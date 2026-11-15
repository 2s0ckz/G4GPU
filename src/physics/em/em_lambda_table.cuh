// The lambda tables of a G4VEmProcess - host only, built once at upload - for the photon's four
// processes, which `G4GammaGeneralProcess::BuildPhysicsTable` reads through `GetLambda` when it
// fills its own tables (P26, docs/RISK.md V245-).
//
// WHY THE PORT NEEDS THEM AT ALL
//
// P1's photon evaluates its models at the photon's energy, and Geant4's does not: the general
// process's zone totals and selection fractions are tables whose nodes are each sub-process's
// `GetLambda(e, couple, loge)` - and `GetLambda` reads that sub-process's OWN tables. Compton's
// "prim" table above 1 MeV is E*sigma on a splined log grid, and near its first node the spline is
// 0.43% under the model (V225). A general-process table built from the models would carry none of
// that; one built from these carries all of it. So this file is G4VEmProcess's half of the job:
//
//   G4VEmProcess::BuildLambdaTable      the bin count from the EM parameters' binning
//   G4EmTableUtil::BuildLambdaTable     the two vectors per couple, their edges and spline flags
//   G4EmModelManager::FillLambdaVector  the node values, one model, `startFromNull`, E*sigma prim
//   G4VEmProcess::GetCurrentLambda      which vector answers at an energy, or the model itself
//
// and the per-process settings each 11.1.1 constructor and `InitialiseProcess` make (the
// `kCompton`/`kConversion`/`kPhotoelectric`/`kRayleigh` rows of `em_process_lambda_config`).
//
// G4PhysicsLogVector's constructor and `LogVectorValue` with the spline are transcribed in
// `LogVecHost`; the spline's second derivatives are `data::fill_second_derivatives`, which is
// `G4PhysicsVector::ComputeSecDerivative1` - what `FillSecondDerivatives()` with its default
// `G4SplineType::Base` runs.
#pragma once
#include <cfloat>
#include <cmath>
#include <vector>
#include "data/g4spline.hh"

namespace g4gpu::em {

/// `G4EmParameters`' defaults as `G4EmStandardPhysics` (option 0) leaves them: `MinKinEnergy`
/// 0.1 keV, `MaxKinEnergy` 100 TeV, `NumberOfBinsPerDecade` 7 (G4EmParameters::SetDefaults,
/// 11.1.1). MeV.
inline constexpr double kEmKeV = 1.e-3;  ///< CLHEP::keV, so that `0.1*keV` is the same double
inline constexpr double kEmParamMinKinEnergy = 0.1 * kEmKeV;
inline constexpr double kEmParamMaxKinEnergy = 1.0e8;
inline constexpr int kEmParamBinsPerDecade = 7;

/// `G4lrint` (templates.hh, 11.1.1): `(int)std::lrint(ad)` - round half to even under the
/// default rounding mode.
inline int g4_lrint(double ad) { return static_cast<int>(std::lrint(ad)); }

/// One `G4PhysicsLogVector` on the host: its constructor's node arithmetic, its values, and the
/// second derivatives when it is splined.
struct LogVecHost {
  std::vector<double> e, v, d2;
  double edge_min = 0, edge_max = 0, inv_dbin = 0, log_emin = 0;
  bool spline = false;

  /// `G4PhysicsLogVector(Emin, Emax, Nbin, spline)`: the edges stored as given, `Initialise()`
  /// (`invdBin = (idxmax + 1)/G4Log(edgeMax/edgeMin)`, `logemin = G4Log(edgeMin)`), then
  /// `binVector[i] = edgeMin*G4Exp(i/invdBin)` inside. `G4Log`/`G4Exp` are `std::log`/`std::exp`
  /// on this platform (G4Log.hh, G4Exp.hh under WIN32), so the nodes are Geant4's to the bit.
  void make(double emin, double emax, int nbin, bool use_spline) {
    int nodes = nbin + 1;
    if (nodes < 3) { nodes = 3; }
    e.assign(static_cast<std::size_t>(nodes), 0.0);
    v.assign(static_cast<std::size_t>(nodes), 0.0);
    d2.clear();
    e[0] = emin;
    e[static_cast<std::size_t>(nodes - 1)] = emax;
    edge_min = emin;
    edge_max = emax;
    inv_dbin = static_cast<double>(nodes - 1) / std::log(edge_max / edge_min);
    log_emin = std::log(edge_min);
    for (int i = 1; i <= nodes - 2; ++i) {
      e[static_cast<std::size_t>(i)] = edge_min * std::exp(static_cast<double>(i) / inv_dbin);
    }
    spline = use_spline;
  }
  int n() const { return static_cast<int>(e.size()); }

  /// `FillSecondDerivatives()` with the default `G4SplineType::Base`: nothing unless the vector
  /// was made with the spline, and the spline switched OFF for fewer than 5 nodes, as Geant4
  /// switches it off.
  void fill_second_derivatives() {
    if (!spline) { return; }
    if (n() < 5) {
      spline = false;
      return;
    }
    d2.assign(e.size(), 0.0);
    data::fill_second_derivatives(e.data(), v.data(), n(), d2.data());
  }

  /// `G4PhysicsVector::LogVectorValue(e, loge)`: the bin from `ComputeLogVectorBin` (truncation
  /// toward zero on a G4int, clamped to idxmax), then `Interpolation` - with the spline term
  /// when the vector has one, in Geant4's order of operations - and the end values outside.
  double log_value(double en, double loge) const {
    if (en > edge_min && en < edge_max) {
      const int idxmax = n() - 2;
      int idx = static_cast<int>((loge - log_emin) * inv_dbin);
      if (idx > idxmax) { idx = idxmax; }
      const double x1 = e[static_cast<std::size_t>(idx)];
      const double dl = e[static_cast<std::size_t>(idx) + 1] - x1;
      const double y1 = v[static_cast<std::size_t>(idx)];
      const double dy = v[static_cast<std::size_t>(idx) + 1] - y1;
      const double b = (en - x1) / dl;
      double res = y1 + b * dy;
      if (spline) {
        const double c0 = (2.0 - b) * d2[static_cast<std::size_t>(idx)];
        const double c1 = (1.0 + b) * d2[static_cast<std::size_t>(idx) + 1];
        res += (b * (b - 1.0)) * (c0 + c1) * (dl * dl * (1.0 / 6.0));
      }
      return res;
    }
    return (en <= edge_min) ? v.front() : v.back();
  }
};

/// What one G4VEmProcess's constructor and `InitialiseProcess` set that its lambda tables depend
/// on.
struct EmProcessLambdaConfig {
  double min_kin = kEmParamMinKinEnergy;      ///< `minKinEnergy`
  double max_kin = kEmParamMaxKinEnergy;      ///< `maxKinEnergy`
  double min_kin_prim = DBL_MAX;              ///< `minKinEnergyPrim`; DBL_MAX is "no prim table"
  int lambda_bins = 0;                        ///< `nLambdaBins` when `actBinning`, else 0
  bool build_lambda = true;                   ///< `buildLambdaTable`
  bool start_from_null = false;               ///< `startFromNull`
  double min_primary_energy = 0.0;            ///< `MinPrimaryEnergy(part, material)`
  bool spline = true;                         ///< `splineFlag`
};

enum class GammaSubProcess : int { kCompton = 0, kConversion = 1, kPhotoelectric = 2, kRayleigh = 3 };

/// The four photon processes of `G4EmStandardPhysics` (11.1.1) as their constructors leave them:
///
///   G4ComptonScattering   SetStartFromNullFlag(true), SetBuildTableFlag(true),
///                         SetMinKinEnergyPrim(1*MeV), SetSplineFlag(true); MinPrimaryEnergy is
///                         G4VEmProcess's 0, so `startFromNull` never takes effect.
///   G4GammaConversion     SetMinKinEnergy(2*m_e) (constructor and InitialiseProcess),
///                         SetStartFromNullFlag(true), SetBuildTableFlag(true),
///                         SetLambdaBinning(220); MinPrimaryEnergy = 2*m_e; spline default true.
///   G4PhotoElectricEffect SetBuildTableFlag(false), SetMinKinEnergyPrim(200*keV): no lambda
///                         vector - the model itself below 200 keV - and a prim vector above.
///   G4RayleighScattering  SetStartFromNullFlag(false), SetBuildTableFlag(true),
///                         SetMinKinEnergyPrim(150*keV), SetSplineFlag(false) - which the prim
///                         vector ignores: G4EmTableUtil always splines it.
inline EmProcessLambdaConfig em_process_lambda_config(GammaSubProcess p, double electron_mass) {
  EmProcessLambdaConfig c;
  switch (p) {
    case GammaSubProcess::kCompton:
      c.min_kin_prim = 1.0;
      c.start_from_null = true;
      c.min_primary_energy = 0.0;
      c.spline = true;
      break;
    case GammaSubProcess::kConversion:
      c.min_kin = 2.0 * electron_mass;
      c.lambda_bins = 220;
      c.start_from_null = true;
      c.min_primary_energy = 2.0 * electron_mass;
      c.spline = true;
      break;
    case GammaSubProcess::kPhotoelectric:
      c.build_lambda = false;
      c.min_kin_prim = 200 * kEmKeV;
      c.spline = true;
      break;
    case GammaSubProcess::kRayleigh:
      c.min_kin_prim = 150 * kEmKeV;
      c.start_from_null = false;
      c.spline = false;
      break;
  }
  return c;
}

/// One G4VEmProcess's tables for one material, and `GetLambda` on them.
struct EmProcessLambda {
  EmProcessLambdaConfig cfg;
  bool has_lambda = false;
  bool has_prim = false;
  LogVecHost lambda;
  LogVecHost prim;

  /// `G4VEmProcess::BuildLambdaTable` and `G4EmTableUtil::BuildLambdaTable` for one couple, with
  /// `xs(e)` the model's `CrossSection(couple, particle, e, cut, tmax)` per volume, 1/mm. One model
  /// over the whole range (each of the four `AddEmModel(1, ...)` once), so `FillLambdaVector`'s
  /// `del` is 0.0 and `(1.0 + del/e)*xs` is `xs` to the bit.
  template <typename XsFn>
  void build(const EmProcessLambdaConfig& c, XsFn xs) {
    cfg = c;
    double scale = kEmParamMaxKinEnergy / kEmParamMinKinEnergy;
    int nbin = kEmParamBinsPerDecade * g4_lrint(std::log10(scale));
    if (c.lambda_bins > 0 && c.lambda_bins > nbin) { nbin = c.lambda_bins; }
    scale = nbin / std::log(scale);
    const double emax1 = (c.max_kin < c.min_kin_prim) ? c.max_kin : c.min_kin_prim;
    has_lambda = c.build_lambda;
    if (has_lambda) {
      double emin = c.min_kin;
      bool start_null = false;
      if (c.start_from_null) {
        const double e = c.min_primary_energy;
        if (e >= emin) {
          emin = e;
          start_null = true;
        }
      }
      double emax = emax1;
      if (emax <= emin) { emax = 2 * emin; }
      int bin = g4_lrint(scale * std::log(emax / emin));
      if (bin < 5) { bin = 5; }
      lambda.make(emin, emax, bin, c.spline);
      for (int j = 0; j < lambda.n(); ++j) {
        double cross = xs(lambda.e[static_cast<std::size_t>(j)]);
        if (j == 0 && start_null) { cross = 0.0; }
        lambda.v[static_cast<std::size_t>(j)] = (cross > 0.0) ? cross : 0.0;
      }
      lambda.fill_second_derivatives();
    }
    has_prim = (c.min_kin_prim < c.max_kin);
    if (has_prim) {
      int bin = g4_lrint(scale * std::log(c.max_kin / c.min_kin_prim));
      if (bin < 5) { bin = 5; }
      prim.make(c.min_kin_prim, c.max_kin, bin, true);
      for (int j = 0; j < prim.n(); ++j) {
        const double e = prim.e[static_cast<std::size_t>(j)];
        double cross = xs(e);
        cross *= e;  // `if(fIsCrossSectionPrim == tType) { cross *= e; }`
        prim.v[static_cast<std::size_t>(j)] = (cross > 0.0) ? cross : 0.0;
      }
      prim.fill_second_derivatives();
    }
  }

  /// `G4VEmProcess::GetLambda(e, couple, loge)` -> `GetCurrentLambda(e, loge)`, `fFactor` 1:
  /// the prim vector over E from `minKinEnergyPrim` up, else the lambda vector, else the model.
  template <typename XsFn>
  double get_lambda(double e, double loge, XsFn xs) const {
    if (e >= cfg.min_kin_prim) { return prim.log_value(e, loge) / e; }
    if (has_lambda) { return lambda.log_value(e, loge); }
    return xs(e);
  }
};

}  // namespace g4gpu::em
