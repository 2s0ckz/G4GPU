// The e+- discrete rates as Geant4 11.1.1 draws them: the lambda tables of `eIoni` and `eBrem`
// and the INTEGRAL APPROACH of G4VEnergyLossProcess and G4VEmProcess (P44, docs/RISK.md V351,
// closing the half of V78 that a lepton uses).
//
// WHAT GEANT4 DOES, and what this port did until P44.
//
// `G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength` does not evaluate the model's
// cross section at the pre-step energy. It reads its LAMBDA TABLE - a `G4PhysicsLogVector` per
// couple, built by `G4EmTableUtil::BuildLambdaTable` from the model on the EM parameters' grid,
// splined - and not at the pre-step energy either: `ComputeLambdaForScaledEnergy` takes the
// largest value the cross section can reach over the energies the step will pass through
// ("preStepLambda"), and `PostStepDoIt` then REJECTS the interaction with probability
// `1 - lambda(E_post)/preStepLambda`:
//
//     // Integral approach                                   G4VEnergyLossProcess.cc:1002
//     if (fXSType != fEmNoIntegral) {
//       G4double lx = GetLambdaForScaledEnergy(postStepScaledEnergy, logFinalT + logMassRatio);
//       lx = std::max(lx, 0.0);
//       if(preStepLambda*G4UniformRand() >= lx) { return &fParticleChange; }
//     }
//
// That is thinning: an interaction process of rate preStepLambda, each point kept with the
// ratio of the true rate there to the majorant, which samples the rate lambda(E(s)) ALONG the
// step rather than lambda(E_pre) over all of it. The port drew `-log(u)/sigma_model(E_pre)`.
// Above the cross section's peak the rate RISES as the particle slows - a Moller cross section
// with a fixed cut goes as ~1/(beta^2 E) - so the pre-step value is the minimum over the step and
// every step under-produced. In a dense material, where the e- cut is 280-460 keV, the discrete
// share of the loss is small and so is the error; in a GAS, where the cut is the 990 eV floor,
// almost all of a 100 keV electron's loss is delta rays above the cut and the error is the dose:
// V350's tape counts 4.1% fewer delta rays in air per 100 keV electron than Geant4, which is the
// air row's -1.5% (V341) together with the boundary deposit of V351's first commit.
//
// WHAT IS TRANSCRIBED, function by function (11.1.1):
//
//   G4VEnergyLossProcess::BuildLambdaTable    scale = 7*lrint(log10(1e8/1e-4))/ln(1e12)
//   G4EmTableUtil::BuildLambdaTable           emin = MinPrimaryEnergy (2*cut for e- eIoni, cut
//                                             for e+ eIoni and for eBrem); startNull when that
//                                             is above minKinEnergy (0.1 keV - always, the cut
//                                             floor is 990 eV); emax 100 TeV; max(bin, 5);
//                                             splined (`spline` defaults to true)
//   G4EmModelManager::FillLambdaVector        the model's CrossSection(e, cut, DBL_MAX); eBrem's
//                                             two models joined at 1 GeV by `(1 + del/e)`, which
//                                             is `data::brems_dedx_xs`'s own boundary factor
//   G4EmUtility::FindCrossSectionMax          eIoni's peak per couple (fEmOnePeak)
//   G4EmUtility::FillPeaksStructure           eBrem's two peaks (fEmTwoPeaks), with
//                                             SetLambdaTable's fall-backs to one peak and to
//                                             fEmIncreasing
//   ComputeLambdaForScaledEnergy              the majorant, below
//   PostStepDoIt's rejection                  `accept`, below
//   G4VEmProcess, fEmDecreasing               e+ annihilation in flight (G4eplusAnnihilation:
//                                             no table, `SetCrossSectionType(fEmDecreasing)`):
//                                             majorant at lambdaFactor*E, rejection at E_post
//
// THE ONE THING NOT CARRIED, AND WHY IT IS THE SAME DISTRIBUTION. Geant4 caches `preStepLambda`
// and `mfpKinEnergy` across a track's steps and recomputes them only when the energy leaves the
// window the cache was made for. Every value the cache can hold is a MAJORANT of the rate over
// the step it is used for, and thinning against any majorant samples the same process - the
// cache decides how many candidate points are thrown away, not how many survive. So the majorant
// here is computed afresh each step, as Geant4 computes it on the first step of a track and after
// every interaction (`mfpKinEnergy = DBL_MAX`), and the interaction length is redrawn each step
// as every other discrete length in this transport is (`TrackState::ngp_left`'s note: exact for a
// process that charges a step at the length it was drawn with, which this one does). The cases
// in which Geant4's own scheme is not exact - a step whose energy falls below the majorant's
// energy, lambdaFactor*E, inside one step - are the cases this one is not exact in either; the
// step function's dRoverRange of 0.2 is what keeps them rare on both sides.
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "data/brems_data.cuh"
#include "data/g4spline.hh"
#include "data/materials.cuh"
#include "physics/em/annihilation.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/em_lambda_table.cuh"

namespace g4gpu::em {

/// Nodes a lambda vector can have. The longest is a vector from the 990 eV cut floor to 100 TeV:
/// `lrint(84/ln(1e12) * ln(1e8/9.9e-4))` = 77 bins, 78 nodes. The builder refuses more by name.
constexpr int kLepLamMaxNodes = 96;

/// `G4EmParameters::LambdaFactor()`, 0.8 by default (printed by every Geant4 run as "X-section
/// factor for integral approach 0.8").
template <typename real_t> __host__ __device__ constexpr real_t kLambdaFactor() {
  return real_t(0.8);
}

/// `G4CrossSectionType`, the four the e+- processes can end up with.
enum class LepXsType : int { kIncreasing = 0, kOnePeak = 1, kTwoPeaks = 2, kDecreasing = 3 };

/// One `G4PhysicsLogVector`, on the device. `LogVecHost` is the host builder of the same object.
template <typename real_t>
struct LepLamVec {
  int n = 0;  ///< nodes; 0 = no vector (no rate)
  bool spline = false;
  real_t edge_min = 0, edge_max = 0, inv_dbin = 0, log_emin = 0;
  real_t e[kLepLamMaxNodes];
  real_t v[kLepLamMaxNodes];
  real_t d2[kLepLamMaxNodes];

  /// `G4PhysicsVector::LogVectorValue(e, loge)` - `LogVecHost::log_value` line for line.
  /// `Value(e)` is the same number: for a log vector `GetBin` is `ComputeLogVectorBin(G4Log(e))`.
  __host__ __device__ real_t value(real_t en, real_t loge) const {
    if (n == 0) { return real_t(0); }
    if (en > edge_min && en < edge_max) {
      const int idxmax = n - 2;
      int idx = static_cast<int>((loge - log_emin) * inv_dbin);
      if (idx > idxmax) { idx = idxmax; }
      const real_t x1 = e[idx];
      const real_t dl = e[idx + 1] - x1;
      const real_t y1 = v[idx];
      const real_t dy = v[idx + 1] - y1;
      const real_t b = (en - x1) / dl;
      real_t res = y1 + b * dy;
      if (spline) {
        const real_t c0 = (real_t(2) - b) * d2[idx];
        const real_t c1 = (real_t(1) + b) * d2[idx + 1];
        res += (b * (b - real_t(1))) * (c0 + c1) * (dl * dl * (real_t(1) / real_t(6)));
      }
      return res;
    }
    return (en <= edge_min) ? v[0] : v[n - 1];
  }
};

/// `G4TwoPeaksXS`.
template <typename real_t>
struct LepTwoPeaks {
  real_t e1peak, e1deep, e2peak, e2deep, e3peak;
};

/// eIoni and eBrem of e- [0] and e+ [1], per material, and their cross-section types.
template <typename real_t>
struct LeptonLambdaTables {
  int n_materials = 0;
  LepLamVec<real_t> ioni[2][data::kMaxMaterials];
  LepLamVec<real_t> brem[2][data::kMaxMaterials];
  LepXsType ioni_type[2];
  LepXsType brem_type[2];
  real_t ioni_epeak[2][data::kMaxMaterials];  ///< `theEnergyOfCrossSectionMax`, DBL_MAX: none
  real_t brem_epeak[2][data::kMaxMaterials];  ///< the same, when eBrem fell back to one peak
  LepTwoPeaks<real_t> brem_peaks[2][data::kMaxMaterials];

  /// `GetLambdaForScaledEnergy`: the vector's value (`fFactor` is 1 without biasing; the mass
  /// ratio of an e+- is 1, so the scaled energy is the kinetic energy).
  __host__ __device__ static real_t lam(const LepLamVec<real_t>& vec, real_t e) {
    return vec.value(e, log(e));
  }

  /// `ComputeLambdaForScaledEnergy` with `mfpKinEnergy = DBL_MAX` - the majorant of a fresh
  /// step (see the header's note on the cache). `loge` is `preStepLogScaledEnergy`.
  __host__ __device__ static real_t majorant(const LepLamVec<real_t>& vec, LepXsType type,
                                             real_t epeak, const LepTwoPeaks<real_t>& pk,
                                             real_t e, real_t loge) {
    const real_t lf = kLambdaFactor<real_t>();
    if (type == LepXsType::kIncreasing) { return vec.value(e, loge); }
    if (type == LepXsType::kOnePeak) {
      if (e <= epeak) { return vec.value(e, loge); }
      return lam(vec, fmax(epeak, e * lf));
    }
    // kTwoPeaks
    if (e <= pk.e1peak) { return vec.value(e, loge); }
    if (e <= pk.e1deep) { return lam(vec, fmax(pk.e1peak, e * lf)); }
    if (e <= pk.e2peak) { return vec.value(e, loge); }
    if (e <= pk.e2deep) { return lam(vec, fmax(pk.e2peak, e * lf)); }
    if (e <= pk.e3peak) { return vec.value(e, loge); }
    return lam(vec, fmax(pk.e3peak, e * lf));
  }

  __host__ __device__ real_t ioni_majorant(int mat, bool pos, real_t e, real_t loge) const {
    const int p = pos ? 1 : 0;
    return majorant(ioni[p][mat], ioni_type[p], ioni_epeak[p][mat], brem_peaks[p][mat], e, loge);
  }
  __host__ __device__ real_t brem_majorant(int mat, bool pos, real_t e, real_t loge) const {
    const int p = pos ? 1 : 0;
    return majorant(brem[p][mat], brem_type[p], brem_epeak[p][mat], brem_peaks[p][mat], e, loge);
  }
  /// `lx` of the rejection: the table at the POST-step energy.
  __host__ __device__ real_t ioni_post(int mat, bool pos, real_t e) const {
    return (e > real_t(0)) ? fmax(lam(ioni[pos ? 1 : 0][mat], e), real_t(0)) : real_t(0);
  }
  __host__ __device__ real_t brem_post(int mat, bool pos, real_t e) const {
    return (e > real_t(0)) ? fmax(lam(brem[pos ? 1 : 0][mat], e), real_t(0)) : real_t(0);
  }
};

/// G4eplusAnnihilation, `fEmDecreasing` and no table: `ComputeIntegralLambda`'s
/// `preStepLambda = GetCurrentLambda(e*lambdaFactor)` - the model, through `ComputeCurrentLambda`.
template <typename real_t>
__host__ __device__ inline real_t annihilation_majorant(const data::Material<real_t>& m,
                                                        real_t e) {
  return annihilation_xs(m, e * kLambdaFactor<real_t>());
}

/// The rejection itself: `preStepLambda*G4UniformRand() >= lx` means NO interaction.
template <typename real_t, typename Rng>
__host__ __device__ inline bool integral_accept(real_t pre_step_lambda, real_t lx, Rng& rng) {
  return !(pre_step_lambda * rng.uniform() >= fmax(lx, real_t(0)));
}

// ------------------------------------------------------------------------------- host builder

/// `G4EmTableUtil::BuildLambdaTable` for one couple of a G4VEnergyLossProcess, into `out`.
/// `xs(e)` is the model's `CrossSection(couple, particle, e, cut, DBL_MAX)`, 1/mm.
template <typename real_t, typename XsFn>
__host__ inline bool build_eloss_lambda_vec(double min_primary_energy, XsFn xs,
                                            LepLamVec<real_t>& out) {
  double scale = kEmParamMaxKinEnergy / kEmParamMinKinEnergy;
  const int nbin = kEmParamBinsPerDecade * g4_lrint(std::log10(scale));
  scale = nbin / std::log(scale);
  bool start_null = true;
  double emin = min_primary_energy;
  if (kEmParamMinKinEnergy > emin) {
    emin = kEmParamMinKinEnergy;
    start_null = false;
  }
  double emax = kEmParamMaxKinEnergy;
  if (emax <= emin) { emax = 2 * emin; }
  int bin = g4_lrint(scale * std::log(emax / emin));
  if (bin < 5) { bin = 5; }
  LogVecHost h;
  h.make(emin, emax, bin, /*spline=*/true);
  if (h.n() > kLepLamMaxNodes) {
    std::printf("FATAL: an e+- lambda vector of %d nodes is past kLepLamMaxNodes = %d\n", h.n(),
                kLepLamMaxNodes);
    return false;
  }
  for (int j = 0; j < h.n(); ++j) {
    double cross = xs(h.e[static_cast<std::size_t>(j)]);
    if (j == 0 && start_null) { cross = 0.0; }
    h.v[static_cast<std::size_t>(j)] = (cross > 0.0) ? cross : 0.0;
  }
  h.fill_second_derivatives();
  out.n = h.n();
  out.spline = h.spline;
  out.edge_min = static_cast<real_t>(h.edge_min);
  out.edge_max = static_cast<real_t>(h.edge_max);
  out.inv_dbin = static_cast<real_t>(h.inv_dbin);
  out.log_emin = static_cast<real_t>(h.log_emin);
  for (int j = 0; j < h.n(); ++j) {
    out.e[j] = static_cast<real_t>(h.e[static_cast<std::size_t>(j)]);
    out.v[j] = static_cast<real_t>(h.v[static_cast<std::size_t>(j)]);
    out.d2[j] = h.spline ? static_cast<real_t>(h.d2[static_cast<std::size_t>(j)]) : real_t(0);
  }
  return true;
}

/// `G4EmUtility::FindCrossSectionMax` over one process's vectors (one per couple). Returns
/// false - Geant4's null pointer - when no vector has a peak.
template <typename real_t>
__host__ inline bool find_cross_section_max(const LepLamVec<real_t>* vecs, int n,
                                            real_t* epeak) {
  bool is_peak = false;
  for (int i = 0; i < n; ++i) {
    epeak[i] = static_cast<real_t>(DBL_MAX);
    const LepLamVec<real_t>& pv = vecs[i];
    double xs = 0.0, ee = 0.0;
    for (int j = 0; j < pv.n; ++j) {
      const double e = pv.e[j];
      const double ss = pv.v[j];
      if (ss >= xs) {
        xs = ss;
        ee = e;
        continue;
      }
      is_peak = true;
      epeak[i] = static_cast<real_t>(ee);
      break;
    }
  }
  return is_peak;
}

/// `G4EmUtility::FillPeaksStructure`, without the base-material loop (B1 and every scene this
/// engine builds has no base materials: `GetBaseMaterialFlag()` is false). Returns false - the
/// null pointer - when no vector has a deep.
template <typename real_t>
__host__ inline bool fill_peaks_structure(const LepLamVec<real_t>* vecs, int n,
                                          LepTwoPeaks<real_t>* out) {
  bool is_deep = false;
  for (int i = 0; i < n; ++i) {
    const LepLamVec<real_t>& pv = vecs[i];
    double ee = 0.0, xs = 0.0;
    double e1peak = DBL_MAX, e1deep = DBL_MAX, e2peak = DBL_MAX, e2deep = DBL_MAX,
           e3peak = DBL_MAX;
    for (int j = 0; j < pv.n; ++j) {
      const double e = pv.e[j];
      const double ss = pv.v[j];
      if (e1peak == DBL_MAX) {
        if (ss >= xs) {
          xs = ss;
          ee = e;
          continue;
        } else {
          e1peak = ee;
        }
      }
      if (e1deep == DBL_MAX) {
        if (ss <= xs) {
          xs = ss;
          ee = e;
          continue;
        } else {
          e1deep = ee;
          is_deep = true;
        }
      }
      if (e2peak == DBL_MAX) {
        if (ss >= xs) {
          xs = ss;
          ee = e;
          continue;
        } else {
          e2peak = ee;
        }
      }
      if (e2deep == DBL_MAX) {
        if (ss <= xs) {
          xs = ss;
          ee = e;
          continue;
        } else {
          e2deep = ee;
          break;
        }
      }
      if (e3peak == DBL_MAX) {
        if (ss >= xs) {
          xs = ss;
          ee = e;
          continue;
        } else {
          e3peak = ee;
        }
      }
    }
    out[i] = LepTwoPeaks<real_t>{static_cast<real_t>(e1peak), static_cast<real_t>(e1deep),
                                 static_cast<real_t>(e2peak), static_cast<real_t>(e2deep),
                                 static_cast<real_t>(e3peak)};
  }
  return is_deep;
}

/// The four processes' tables for every material of the scene. `sb` must hold the
/// Seltzer-Berger tables of every element of `mats` (the brems model needs them).
template <typename real_t>
__host__ inline bool build_lepton_lambda_tables(const data::Material<real_t>* mats, int n,
                                                const data::SBTableSet<real_t>& sb,
                                                LeptonLambdaTables<real_t>& out) {
  out.n_materials = n;
  for (int p = 0; p < 2; ++p) {
    const bool pos = (p == 1);
    for (int m = 0; m < n; ++m) {
      const data::Material<real_t>& mat = mats[m];
      // eIoni: the secondary is an electron for both species, so the electron cut.
      const double ecut = static_cast<double>(mat.cut_electron);
      const double ioni_min = pos ? ecut : 2.0 * ecut;  // G4eIonisation::MinPrimaryEnergy
      if (!build_eloss_lambda_vec<real_t>(
              ioni_min,
              [&](double e) {
                return static_cast<double>(
                    delta_ray_xs<real_t>(mat, static_cast<real_t>(e), pos, mat.cut_electron));
              },
              out.ioni[p][m])) {
        return false;
      }
      // eBrem: G4VEnergyLossProcess::MinPrimaryEnergy, the gamma cut.
      const data::BremsBoundary<real_t> bnd = data::brems_boundary<real_t>(mat, sb, pos);
      if (!build_eloss_lambda_vec<real_t>(
              static_cast<double>(mat.cut_gamma),
              [&](double e) {
                real_t dedx = 0, xs = 0;
                data::brems_dedx_xs<real_t>(mat, sb, bnd, static_cast<real_t>(e), pos, dedx, xs);
                return static_cast<double>(xs);
              },
              out.brem[p][m])) {
        return false;
      }
    }
    // eIoni: fEmOnePeak (G4VEnergyLossProcess's default), and fEmIncreasing if no couple peaks.
    out.ioni_type[p] = find_cross_section_max<real_t>(out.ioni[p], n, out.ioni_epeak[p])
                           ? LepXsType::kOnePeak
                           : LepXsType::kIncreasing;
    // eBrem: fEmTwoPeaks; SetLambdaTable falls back to one peak, then to increasing.
    if (fill_peaks_structure<real_t>(out.brem[p], n, out.brem_peaks[p])) {
      out.brem_type[p] = LepXsType::kTwoPeaks;
      for (int m = 0; m < n; ++m) { out.brem_epeak[p][m] = static_cast<real_t>(DBL_MAX); }
    } else {
      out.brem_type[p] = find_cross_section_max<real_t>(out.brem[p], n, out.brem_epeak[p])
                             ? LepXsType::kOnePeak
                             : LepXsType::kIncreasing;
    }
  }
  return true;
}

}  // namespace g4gpu::em
