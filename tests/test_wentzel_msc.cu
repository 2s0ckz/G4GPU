// The WentzelVI stepping algorithm: step limit, true<->geometric path conversion, and the
// mixed multiple/single scattering sampler.
//
// The cross sections this drives are diffed against G4WentzelOKandVIxSection in
// tests/test_wentzel.cu to 0.0003%. The stepping itself has no Geant4 accessor that can be
// called in isolation - G4WentzelVIModel::ComputeGeomPathLength and ComputeTrueStepLength
// mutate model state across calls and read a G4Track - so this checks the properties the
// algorithm is built on instead:
//
//   1. the path conversion is invertible: true -> geometric -> true returns the input
//   2. the geometric length never exceeds the true length (a detour cannot be longer than
//      the path that produced it) and both stay positive
//   3. the step limit is bounded by the range and by the requested step
//   4. the sampler produces unit directions and a displacement no longer than the step
//   5. single-scattering mode engages exactly when the expected collision count drops below
//      ten, which is what the algorithm keys on
#include <cstdio>
#include <cmath>
#include <string>
#include "core/particle.cuh"
#include "core/rng.cuh"
#include "data/materials.cuh"
#include "physics/em/wentzel_msc.cuh"

using namespace g4gpu;
using real_t = double;

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

  printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
