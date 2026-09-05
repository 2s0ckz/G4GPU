// Urban MSC. There is no Geant4 table to diff this against, so these are internal
// consistency checks on the transcription rather than an external validation.
//
// The strong check: G4UrbanMscModel::SampleCosineTheta is *constructed* so that the sampled
// <cos(theta)> equals xmeanth = exp(-tau), the transport-theory mean. It reaches that by
// mixing a core branch (mean xmean1), a tail branch (xmean2) and an isotropic branch
// (mean 0), with the isotropic weight 1-qprob chosen to make the total come out right.
//
// That construction only works when qprob <= 1. Geant4 does not clamp it, so when
// qprob > 1 the isotropic branch never fires and the achieved mean is xmeanth/qprob. That
// is Geant4's behaviour, not a porting artefact, so the invariant asserted here is
//
//     <cos(theta)> == xmeanth / max(1, qprob)
//
// which is exact in every regime. It pins down tau, theta0, the xsi polynomial, xmean1,
// xmean2, prob, qprob and both branch samplers at once: any transcription error in any of
// them moves the sampled mean off the predicted one.
#include <cstdio>
#include <cmath>
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/urban_msc.cuh"

using namespace g4gpu;
using real_t = double;

int main() {
  data::Material<real_t> mats[data::kNumMaterials];
  data::build_b1_materials<real_t>(mats);
  const char* names[4] = {"air", "water", "A-150 tissue", "bone"};
  int fails = 0;

  printf("== transport mean free path, mm ==\n");
  printf("  %-14s %12s %12s %12s %12s\n", "material", "0.1 MeV", "1 MeV", "6 MeV", "20 MeV");
  for (int m = 0; m < data::kNumMaterials; ++m) {
    printf("  %-14s", names[m]);
    for (real_t e : {real_t(0.1), real_t(1.0), real_t(6.0), real_t(20.0)}) {
      const real_t l = em::urban_lambda(mats[m], e, false);
      printf(" %12.5g", l);
      if (!(l > 0) || !std::isfinite(l)) { ++fails; }
    }
    printf("\n");
  }

  printf("\n== Urban coefficients (from Zeff) ==\n");
  printf("  %-14s %8s %10s %10s %10s %10s %10s %10s\n", "material", "Zeff", "coeffth1",
         "coeffth2", "coeffc1", "coeffc2", "coeffc3", "coeffc4");
  for (int m = 0; m < data::kNumMaterials; ++m) {
    const auto c = em::urban_coeffs(mats[m]);
    printf("  %-14s %8.3f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n", names[m], mats[m].z_eff,
           c.coeffth1, c.coeffth2, c.coeffc1, c.coeffc2, c.coeffc3, c.coeffc4);
  }

  printf("\n== theta0, water: e-, e+ (with the posa..pose correction), and Highland ==\n");
  printf("  %10s %10s %12s %12s %12s %10s\n", "E (MeV)", "step (mm)", "e- Urban", "e+ Urban",
         "Highland", "e+/e-");
  for (real_t e : {real_t(0.5), real_t(1.0), real_t(6.0)}) {
    for (real_t st : {real_t(0.1), real_t(1.0)}) {
      const auto c = em::urban_coeffs(mats[data::kWater]);
      const real_t u = em::urban_theta0(mats[data::kWater], c, st, e, e, false);
      const real_t up = em::urban_theta0(mats[data::kWater], c, st, e, e, true);
      const real_t h = em::msc_theta0(mats[data::kWater], e, st);
      printf("  %10.3g %10.3g %12.6f %12.6f %12.6f %10.4f\n", e, st, u, up, h,
             (u > 0) ? up / u : 0.0);
      if (!(u > 0) || !std::isfinite(u) || !(up > 0) || !std::isfinite(up)) { ++fails; }
    }
  }

  printf("\n== <cos(theta)> vs the construction's own target ==\n");
  printf("  %-13s %6s %6s %7s %7s %9s %11s %11s %8s %7s\n", "material", "E", "tau", "theta0",
         "qprob", "branch", "sampled", "predicted", "ratio", "sigma");
  const int N = 400000;
  real_t worst = 0;
  for (int m = 1; m < data::kNumMaterials; ++m) {
    for (real_t e : {real_t(0.5), real_t(2.0), real_t(6.0)}) {
      const auto c = em::urban_coeffs(mats[m]);
      const real_t lam = em::urban_lambda(mats[m], e, false);
      for (real_t tw : {real_t(0.02), real_t(0.1), real_t(0.3), real_t(1.0), real_t(2.0)}) {
        const real_t step = tw * lam;
        em::UrbanDebug<real_t> d{};
        {  // one throwaway call to capture the branch and its parameters
          Philox<real_t> probe(1u, 2u);
          em::urban_sample_cos_theta(mats[m], c, step, e, e, lam, false, probe, real_t(-1), &d);
        }
        Philox<real_t> rng(0x51ed5eedu + m * 977u, unsigned(e * 13) * 31u + unsigned(tw * 100));
        real_t sum = 0, sum2 = 0;
        for (int i = 0; i < N; ++i) {
          // Energy held fixed, so xmeanth is exp(-tau) exactly.
          const real_t v = em::urban_sample_cos_theta(mats[m], c, step, e, e, lam, false, rng);
          sum += v;
          sum2 += v * v;
        }
        const real_t got = sum / N;
        // At large tau cos spreads over most of [-1,1], so the standard error on the mean is
        // itself ~1% of it. Judge the agreement in units of that error, not in percent.
        const real_t stderr_ = std::sqrt(std::max(real_t(0), sum2 / N - got * got) / N);
        const real_t want = d.xmeanth / std::max(real_t(1), d.qprob);
        const real_t nsig = (stderr_ > 0) ? std::fabs(got - want) / stderr_ : real_t(0);
        const real_t r = got / want;
        printf("  %-13s %6.3g %6.3g %7.4f %7.4f %9s %11.6f %11.6f %8.4f %7.1f\n", names[m], e,
               tw, d.theta0, d.qprob, d.branch == 0 ? "main" : "fallback", got, want, r, nsig);
        worst = std::max(worst, nsig);
      }
    }
  }
  printf("\n  worst deviation from the construction's target: %.1f standard errors\n", worst);
  // 45 comparisons, so the largest of 45 standard normals sits near 2.7 sigma on its own;
  // 5 sigma is a real-signal threshold rather than a noise threshold.
  if (worst > 5.0) { printf("  FAIL: exceeds 5 sigma\n"); ++fails; }

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
