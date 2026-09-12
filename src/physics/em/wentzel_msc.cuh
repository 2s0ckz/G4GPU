// The WentzelVI multiple-scattering stepping algorithm, transcribed from G4WentzelVIModel
// (11.1.1): step limitation, the true<->geometric path conversion, and the mixed
// single/multiple scattering sampler.
//
// G4EmStandardPhysics uses this for e-/e+ above 100 MeV and for every charged hadron at all
// energies, paired with G4CoulombScattering for the large-angle tail. It works differently
// from Urban: rather than sampling one deflection from a fitted distribution, it splits the
// step into a small number of "multiple scattering" sub-steps below a cut-off angle plus an
// explicit sequence of single Coulomb scatters above it, and switches to pure
// single-scattering mode when the expected number of collisions falls below ten.
//
// The cross sections this drives are in wentzel_xs.cuh and are validated against
// G4WentzelOKandVIxSection to 0.0003%. The single-scattering SAMPLER below is validated as a
// distribution against Geant4's own, in this model's configuration of it:
// ref/dump/dump_wentzel_msc.cc draws 400,000 SampleSingleScattering angles per cell into
// ref/oracle/wentzel_msc_sample.csv and tests/test_wentzel_msc.cu draws the same number from
// wv_sample_single. That comparison is what docs/RISK.md V47's missing 1/(1 + z1*factD) factor
// had nothing to fail against - the properties checked further down this file all passed
// without it.
//
// CALL ORDER MATTERS. The sequence is wv_step_limit -> wv_geom_path -> (geometry decides the
// actual geometric step) -> wv_true_path -> wv_sample_scattering. wv_true_path is what lowers
// cos_theta_min from 1 and recomputes xtsec above it; skipping it leaves xtsec at the full
// single-scattering cross section, and the sampler then tries to generate millions of
// discrete scatters per step instead of a handful.
//
// Not transcribed: the second-moment correction, which does not exist to transcribe -
// G4WentzelOKandVIxSection::ComputeSecondTransportMoment returns 0.0 in 11.1.1, so
// useSecondMoment changes nothing even when set (docs/RISK.md O10) - and the
// fUseDistanceToBoundary geometry limit, which is live but only when the stepping algorithm
// selects it and WentzelVI's default is fUseSafety.
#pragma once
#include <cmath>
#include <vector>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4spline.hh"
#include "data/materials.cuh"
#include "physics/em/urban_msc.cuh"   // urban_gauss, kFacRange, kFacSafety
#include "data/mott.hh"
#include "physics/em/wentzel_xs.cuh"

namespace g4gpu::em {

// ------------------------------------------------- the e+- transport mean free path table
//
// `G4VMscModel::GetTransportMeanFreePath` READS A TABLE FOR e- AND e+ AND EVALUATES THE MODEL
// FOR AN ION, and the one line that decides is `G4VMscModel::GetParticleChangeForMSC`
// (G4VMscModel.cc:94-95):
//
//     if(p->GetParticleName() != "GenericIon" &&
//        (p->GetPDGMass() < CLHEP::GeV || ForceBuildTableFlag()) ) { ...build xSectionTable... }
//
// docs/PORTED.md 4.4 draws the same distinction for Urban. An electron is 0.511 MeV, so it
// passes, and `xSectionTable` exists for its WentzelVI model as well as for its Urban one.
//
// THE TABLE'S GRID IS THE MODEL'S OWN ENERGY WINDOW AND NOT G4EmParameters' WHOLE RANGE.
// `BuildTableForModel` (G4LossTableBuilder.cc) takes
//
//     emin = max(LowEnergyLimit(), LowEnergyActivationLimit()), then max(emin, MinKinEnergy)
//     emax = min(HighEnergyLimit(), HighEnergyActivationLimit()), then min(emax, MaxKinEnergy)
//     n    = NumberOfBinsPerDecade() * lrint(log10(emax/tmin))
//
// and `G4EmStandardPhysics::ConstructProcess` calls `msc2->SetLowEnergyLimit(MscEnergyLimit())`
// on the WentzelVI model it hands `ConstructElectronMscProcess`. So emin is 100 MeV, emax is
// G4VEmModel's default 100 TeV, and n is 7 * lrint(log10(1e6)) = 42 - forty-three nodes, from
// exactly the energy the Urban model stops at. Splined: `G4VMscModel::useSpline` is true.
//
// WHAT IS STORED IS NOT THE CROSS SECTION AND NOT THE MEAN FREE PATH.
// `BuildTableForModel` fills `model->Value(couple, part, E)`, which is
// `G4VEmModel::Value` = `pFactor * E*E * CrossSectionPerVolume(mat, p, E, 0.0, DBL_MAX)`, and
// `GetTransportMeanFreePath` divides by `E*E` again. Storing E^2 sigma rather than sigma is
// what makes the spline behave: sigma falls as 1/E^2 at these energies, so the tabulated
// quantity is nearly flat over six decades and a cubic through 43 points of it is nearly
// exact. Tabulating sigma itself would be a different interpolation of a function that spans
// twelve orders of magnitude.
//
// AND THE CUT IS ZERO, WHICH IS NOT THE CUT THE REST OF THE MODEL USES.
// `G4VEmModel::CrossSectionPerVolume(mat, p, E, emin, emax)` passes its `emin` straight to
// `ComputeCrossSectionPerAtom` as `cutEnergy`, and `Value` calls it with `0.0` - so
// `G4WentzelVIModel::ComputeCrossSectionPerAtom` does `wokvi->SetupTarget(Z, 0.0)` and
// `G4WentzelOKandVIxSection::ComputeMaxElectronScattering(0)` leaves `cosTetMaxElec` at 1.
// The electron-scattering channel is therefore CLOSED in the tabulated transport cross
// section, and open in `ComputeTransportXSectionPerVolume`, which reads
// `(*currentCuts)[currentMaterialIndex]` - the material's electron production cut - for the
// `xtsec` the single-scattering sampler is driven by. Two cross sections, two cuts, one model.
// Passing the production cut here instead adds the electron term to lambda_eff and lengthens
// every step limit; `tests/test_electron_hi.cu` measures what that is worth.
//
// The hadron path in `step_hadron` still calls `wentzel_lambda` directly with the production
// cut, which is neither of these two things. That is left exactly as it is - the proton and
// alpha B1 rows are checked to 0.25% and this package must not move them - and recorded in
// docs/RISK.md V79.

/// Forty-three nodes from `G4EmParameters::MscEnergyLimit()` to `MaxKinEnergy`.
constexpr int kWvLeptonBins = 43;
constexpr double kWvLeptonEMin = 1e2;   ///< MeV - MscEnergyLimit(), the Urban/WentzelVI split
constexpr double kWvLeptonEMax = 1e8;   ///< MeV - G4EmParameters::MaxKinEnergy
constexpr int kWvLeptonBinsPerDecade = 7;

/// `G4VMscModel::xSectionTable` for the e+- WentzelVI model, one row per species and material.
template <typename real_t>
struct WentzelLeptonTable {
  real_t e_min, e_max;  ///< MeV
  int n_materials = data::kNumMaterials;
  real_t energy[kWvLeptonBins];
  /// E^2 * transport cross section per volume, MeV^2/mm. See the header block.
  real_t e2xs[2][data::kMaxMaterials][kWvLeptonBins];
  real_t e2xs_d2[2][data::kMaxMaterials][kWvLeptonBins];

  /// `G4VMscModel::GetTransportMeanFreePath`, mm. Returns a huge length where the cross
  /// section is zero, which is Geant4's `DBL_MAX`.
  __host__ __device__ real_t lambda_at(int material, bool pos, real_t kinetic) const {
    const int p = pos ? 1 : 0;
    const real_t v = data::spline_value<real_t>(energy, e2xs[p][material],
                                                e2xs_d2[p][material], kWvLeptonBins, kinetic);
    const real_t x = v / (kinetic * kinetic);
    return (x > real_t(0)) ? real_t(1) / x : real_t(1e30);
  }
};

/// Fills the table. Host only, once, at start-up.
template <typename real_t>
__host__ inline void build_wentzel_lepton_table(const data::Material<real_t>* mats,
                                                int n_materials,
                                                WentzelLeptonTable<real_t>& t) {
  t.e_min = real_t(kWvLeptonEMin);
  t.e_max = real_t(kWvLeptonEMax);
  t.n_materials = n_materials;
  for (int b = 0; b < kWvLeptonBins; ++b) {
    t.energy[b] =
        static_cast<real_t>(kWvLeptonEMin * std::pow(10.0, double(b) / kWvLeptonBinsPerDecade));
  }

  std::vector<double> x(kWvLeptonBins), y(kWvLeptonBins), d2(kWvLeptonBins);
  for (int b = 0; b < kWvLeptonBins; ++b) { x[b] = static_cast<double>(t.energy[b]); }

  // MscThetaLimit() is pi in option0, so cosThetaLimit is -1 - the same constant
  // `step_lepton` and `step_hadron` pass. The cut is zero; see the header block.
  constexpr real_t kCosThetaLim = real_t(-1);
  for (int p = 0; p < 2; ++p) {
    const ParticleType type = (p == 1) ? ParticleType::kPositron : ParticleType::kElectron;
    const ParticleDef<real_t> pd = particle_def<real_t>(type);
    for (int m = 0; m < n_materials; ++m) {
      for (int b = 0; b < kWvLeptonBins; ++b) {
        const real_t e = t.energy[b];
        const real_t xs =
            wentzel_transport_xs(mats[m], type, pd, e, real_t(0), kCosThetaLim);
        y[b] = static_cast<double>(e) * static_cast<double>(e) * static_cast<double>(xs);
      }
      data::fill_second_derivatives(x.data(), y.data(), kWvLeptonBins, d2.data());
      for (int b = 0; b < kWvLeptonBins; ++b) {
        t.e2xs[p][m][b] = static_cast<real_t>(y[b]);
        t.e2xs_d2[p][m][b] = static_cast<real_t>(d2[b]);
      }
    }
  }
}

// ---------------------------------------------------------------- model constants

/// G4WentzelVIModel::numlimit, the small-tau series cut.
template <typename real_t> __host__ __device__ constexpr real_t kWvNumLimit() {
  return real_t(0.1);
}
/// Below this many expected collisions the model drops to pure single scattering.
constexpr int kWvMinNCollisions = 10;
/// G4WentzelVIModel::tlimitminfix.
template <typename real_t> __host__ __device__ constexpr real_t kWvTlimitMinFix() {
  return real_t(1e-6);  // mm
}
/// SetSingleScatteringFactor(1.25): ssFactor and 1/(ssFactor - 0.05).
template <typename real_t> __host__ __device__ constexpr real_t kWvSsFactor() {
  return real_t(1.25);
}
template <typename real_t> __host__ __device__ constexpr real_t kWvInvSsFactor() {
  return real_t(1) / (real_t(1.25) - real_t(0.05));
}
/// 0.5 pi alpha, the spin-correction coefficient in the single-scattering rejection.
template <typename real_t> __host__ __device__ inline real_t wv_fact_b1() {
  return real_t(0.5) * real_t(3.14159265358979323846) * units::fine_structure_const<real_t>();
}

/// State carried between the limit, the conversion and the sampler - the member variables
/// G4WentzelVIModel keeps on itself.
template <typename real_t>
struct WentzelMscState {
  real_t t_path;       ///< true path length
  real_t z_path;       ///< geometric path length
  real_t lambda_eff;   ///< transport mean free path at the effective energy
  real_t cos_theta_min;///< boundary between multiple and single scattering
  real_t cos_tet_max_nuc;
  real_t xtsec;        ///< total single-scattering cross section above cos_theta_min, 1/mm
  real_t range;
  real_t pre_kin_energy;
  real_t eff_kin_energy;
  bool single_scattering_mode;
};

/// Per-element single-scattering cross sections, needed to pick which atom a discrete
/// scatter happened on. G4WentzelVIModel keeps these in xsecn[] and prob[].
template <typename real_t>
struct WentzelElementXs {
  real_t cumulative[data::kMaxElements];  ///< running sum of the nuclear+electron xs
  real_t electron_fraction[data::kMaxElements];
  int n;
};

/// Transport cross section per volume above @p cos_theta, filling the per-element tables.
/// Verbatim from G4WentzelVIModel::ComputeTransportXSectionPerVolume.
template <typename real_t>
__host__ __device__ inline real_t wv_transport_xs(const data::Material<real_t>& m,
                                                  const ParticleDef<real_t>& pd,
                                                  ParticleType type, real_t kinetic,
                                                  real_t cut, real_t cos_theta_lim,
                                                  real_t cos_theta, real_t cos_tet_max_nuc,
                                                  WentzelElementXs<real_t>& els,
                                                  real_t& xtsec) {
  xtsec = real_t(0);
  els.n = m.n_elements;
  if (cos_tet_max_nuc >= cos_theta) { return real_t(0); }
  real_t xs = real_t(0);
  for (int i = 0; i < m.n_elements; ++i) {
    const int z = static_cast<int>(m.z[i] + real_t(0.5));
    const WentzelState<real_t> s =
        wentzel_setup(pd, type, kinetic, m.inv_a23, z, cut, cos_theta_lim);
    const real_t costm = s.cos_tet_max_nuc;
    real_t esec = real_t(0);
    if (costm < cos_theta) {
      if (cos_theta < real_t(1)) {
        xs += m.n_atoms[i] * wentzel_transport_xs_per_atom(s, z, cos_theta);
      }
      real_t nucsec = wentzel_nuclear_xs(s, z, cos_theta, costm);
      esec = wentzel_electron_xs(s, cos_theta, costm);
      nucsec += esec;
      if (nucsec > real_t(0)) { esec /= nucsec; }
      xtsec += nucsec * m.n_atoms[i];
    }
    els.cumulative[i] = xtsec;
    els.electron_fraction[i] = esec;
  }
  return xs;
}

/// Step limitation, transcribed from G4WentzelVIModel::ComputeTruePathLengthLimit.
/// The fUseDistanceToBoundary geometry term is omitted (B1 selects fUseSafety), which only
/// ever loosens the limit. Note that this is the *only* place the stepping algorithm enters:
/// fMinimal and fUseSafety take the identical path here, so a hadron - for which
/// G4EmParameters sets fMinimal rather than the electron's fUseSafety - needs no separate
/// branch. It does need a different @p fac_range.
///
/// @param fac_range G4VMscModel::facrange, which G4EmTableUtil::PrepareMscProcess fills from
///                  MscRangeFactor (0.04) for a particle lighter than a MeV and
///                  MscMuHadRangeFactor (0.2) for anything heavier. It is a parameter rather
///                  than kFacRange because getting it wrong is a factor of five on the step
///                  length of every proton, and a default here would let that happen quietly.
/// @param lat_off   optional out-parameter, set true on either of the two early returns
///                  below - and `ComputeTruePathLengthLimit` sets `latDisplasment = false` on
///                  both of them, so a caller whose species HAS lateral displacement needs to
///                  know which return it took. Defaulted to null so that `step_hadron`, for
///                  which `MuHadLateralDisplacement` is false and the displacement is
///                  identically zero, is bit-identical to before this parameter existed.
template <typename real_t>
__host__ __device__ inline real_t wv_step_limit(const data::Material<real_t>& m,
                                                const ParticleDef<real_t>& pd,
                                                ParticleType type, real_t kinetic,
                                                real_t range, real_t lambda_eff,
                                                real_t cos_tet_max_nuc, real_t cos_theta_lim,
                                                real_t safety, real_t range_cut,
                                                real_t requested, real_t fac_range,
                                                bool* lat_off = nullptr) {
  real_t tlimit = fmin(requested, range);
  if (tlimit < kWvTlimitMinFix<real_t>()) {
    if (lat_off != nullptr) { *lat_off = true; }
    return tlimit;
  }
  if (range < safety) {
    if (lat_off != nullptr) { *lat_off = true; }
    return tlimit;
  }

  real_t rlimit = fmax(fac_range * range,
                       (real_t(1) - cos_tet_max_nuc) * lambda_eff * kWvInvSsFactor<real_t>());
  if (cos_theta_lim > cos_tet_max_nuc) {
    rlimit = fmin(rlimit, kFacSafety<real_t>() * safety);
  }
  // The production cut expressed as a range softens the limit rather than capping it.
  if (range_cut > rlimit) { rlimit = fmin(rlimit, range_cut * sqrt(rlimit / range_cut)); }
  tlimit = fmin(tlimit, rlimit);
  tlimit = fmax(tlimit, kWvTlimitMinFix<real_t>());
  // facgeom defaults to 2.5 (G4VMscModel).
  tlimit = fmin(tlimit, real_t(50) * m.radiation_length / real_t(2.5));
  return tlimit;
}

/// True -> geometric path length, transcribed from G4WentzelVIModel::ComputeGeomPathLength.
/// Also decides whether the step runs in single-scattering mode.
///
/// @param energy_after kinetic energy left after the whole true step (0 if it stops)
/// @param lambda_at_eff transport mfp at the mean of the pre- and post-step energies
template <typename real_t>
__host__ __device__ inline real_t wv_geom_path(WentzelMscState<real_t>& st, real_t true_length,
                                               real_t energy_after, real_t lambda_at_eff,
                                               real_t cos_tet_max_nuc_eff) {
  st.z_path = st.t_path = true_length;
  st.cos_theta_min = real_t(1);
  // The caller has already filled st.xtsec for cos_theta_min = 1.
  if (st.lambda_eff <= real_t(0)
      || static_cast<int>(st.z_path * st.xtsec) < kWvMinNCollisions) {
    st.single_scattering_mode = true;
    st.lambda_eff = real_t(1e30);
    return st.z_path;
  }
  if (st.t_path < kWvNumLimit<real_t>() * st.lambda_eff) {
    const real_t tau = st.t_path / st.lambda_eff;
    st.z_path *= (real_t(1) - real_t(0.5) * tau + tau * tau / real_t(6));
  } else {
    st.eff_kin_energy = real_t(0.5) * (energy_after + st.pre_kin_energy);
    st.cos_tet_max_nuc = cos_tet_max_nuc_eff;
    st.lambda_eff = lambda_at_eff;
    st.z_path = st.lambda_eff;
    if (st.t_path * kWvNumLimit<real_t>() < st.lambda_eff) {
      st.z_path *= (real_t(1) - exp(-st.t_path / st.lambda_eff));
    }
  }
  return st.z_path;
}

/// Geometric -> true path length, transcribed from
/// G4WentzelVIModel::ComputeTrueStepLength, including the second half that recomputes the
/// scattering cut-off angle and the single-scattering cross section above it.
///
/// @param recompute a callback (material, energy) -> transport cross section per volume at
///                  the new cos_theta_min, which also refreshes the per-element tables
template <typename real_t, typename Recompute>
__host__ __device__ inline real_t wv_true_path(WentzelMscState<real_t>& st, real_t geom_step,
                                               real_t energy_after, real_t lambda_at_eff,
                                               real_t cos_tet_max_nuc_eff,
                                               Recompute recompute) {
  if (st.single_scattering_mode) {
    st.z_path = st.t_path = geom_step;
  } else if (geom_step < st.z_path) {
    if (static_cast<int>(geom_step * st.xtsec) < kWvMinNCollisions) {
      st.z_path = st.t_path = geom_step;
      st.lambda_eff = real_t(1e30);
      st.single_scattering_mode = true;
    } else {
      if (geom_step < kWvNumLimit<real_t>() * st.lambda_eff) {
        const real_t tau = geom_step / st.lambda_eff;
        st.t_path = geom_step * (real_t(1) + real_t(0.5) * tau + tau * tau / real_t(3));
      } else {
        st.t_path *= geom_step / st.z_path;
        st.eff_kin_energy = real_t(0.5) * (energy_after + st.pre_kin_energy);
        st.cos_tet_max_nuc = cos_tet_max_nuc_eff;
        st.lambda_eff = lambda_at_eff;
        const real_t tau = geom_step / st.lambda_eff;
        st.t_path = (tau < real_t(0.999999)) ? -st.lambda_eff * log(real_t(1) - tau)
                                             : st.range;
      }
      st.z_path = geom_step;
    }
  }

  if (!st.single_scattering_mode) {
    st.cos_theta_min -= kWvSsFactor<real_t>() * st.t_path / st.lambda_eff;
    st.xtsec = real_t(0);
    if (st.cos_theta_min > st.cos_tet_max_nuc) {
      const real_t cross = recompute(st.cos_theta_min, st.xtsec);
      if (cross <= real_t(0)) {
        st.single_scattering_mode = true;
        st.t_path = st.z_path;
        st.lambda_eff = real_t(1e30);
        st.cos_theta_min = real_t(1);
      } else if (st.xtsec > real_t(0)) {
        st.lambda_eff = real_t(1) / cross;
        const real_t tau = st.z_path * cross;
        if (tau < kWvNumLimit<real_t>()) {
          st.t_path = st.z_path * (real_t(1) + real_t(0.5) * tau + tau * tau / real_t(3));
        } else if (tau < real_t(0.999999)) {
          st.t_path = -st.lambda_eff * log(real_t(1) - tau);
        } else {
          st.t_path = st.range;
        }
      }
    }
  }
  st.t_path = fmin(st.t_path, st.range);
  return st.t_path;
}

/// The target mass `factD` is built from, MeV.
///
/// Verbatim from G4WentzelOKandVIxSection::SetupTarget (G4WentzelOKandVIxSection.cc:206-208):
///
///     G4double massT = (1 == Z) ? CLHEP::proton_mass_c2 :
///       fNistManager->GetAtomicMassAmu(Z)*CLHEP::amu_c2;
///     SetTargetMass(massT);
///
/// The ELEMENT's mean atomic mass, not an isotope's: this is the one a multiple-scattering step
/// samples with, because G4WentzelVIModel::SampleScattering calls SetupTarget
/// (G4WentzelVIModel.cc:615) and then SampleSingleScattering (:618) with nothing in between. It
/// is deliberately NOT what `em::coulomb_scattering.cuh` uses - G4eCoulombScatteringModel::
/// SampleSecondaries overrides it with the sampled isotope's nuclear mass after the cross
/// sections and before the sampler, so that file takes the mass as an argument and this one
/// derives it from Z. Two callers, two masses, each named where it is chosen.
///
/// `Z`, not `min(Z, 99)`: SetupTarget clamps `targetZ` and then indexes the mass table with the
/// unclamped argument. `data::atomic_mass` returns 0 above Z = 98, exactly as
/// G4NistElementBuilder::GetAtomicMassAmu does, so both sides would divide by zero there; no
/// material can reach this with such a Z, because `data::build_material` divides by the same
/// number to get the atom density.
template <typename real_t>
__host__ __device__ inline real_t wv_target_mass(int z) {
  return (1 == z) ? units::proton_mass_c2<real_t>()
                  : data::atomic_mass<real_t>(z) * units::amu_c2<real_t>();
}

/// One single Coulomb scatter off a chosen element.
///
/// Verbatim from G4WentzelOKandVIxSection::SampleSingleScattering, with the exponential
/// nuclear form factor (G4EmParameters defaults NuclearFormfactorType to fExponentialNF).
/// The rejection function is Geant4's own for both species it distinguishes: the Mott/
/// Rutherford ratio for e+-, the analytic Rutherford-plus-spin expression otherwise.
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> wv_sample_single(const WentzelState<real_t>& s, int z,
                                                         real_t cos_t_min, real_t cos_t_max,
                                                         real_t elec_ratio, Rng& rng) {
  Vec3<real_t> out{real_t(0), real_t(0), real_t(1)};
  real_t formf = s.form_fact_a;
  real_t cost1 = cos_t_min, cost2 = cos_t_max;
  if (elec_ratio > real_t(0) && rng.uniform() <= elec_ratio) {
    formf = real_t(0);  // scattering off an atomic electron: no nuclear form factor
    cost1 = fmax(cost1, s.cos_tet_max_elec);
    cost2 = fmax(cost2, s.cos_tet_max_elec);
  }
  if (cost1 <= cost2) { return out; }

  const real_t w1 = real_t(1) - cost1 + s.screen_z;
  const real_t w2 = real_t(1) - cost2 + s.screen_z;
  const real_t z1 = w1 * w2 / (w1 + rng.uniform() * (w2 - w1)) - s.screen_z;
  real_t fm = real_t(1) + formf * z1;
  fm = real_t(1) / (fm * fm);
  // The rejection function. For e- and e+, G4WentzelOKandVIxSection uses
  // G4ScreeningMottCrossSection's Mott/Rutherford ratio; for everything else the analytic
  // Rutherford-plus-spin expression below. The ratio's beta is the projectile-nucleus
  // relative-system one, which depends on the target Z - so it is computed here, per element,
  // as Geant4 recomputes it with SetupKinematic(tkin, targetZ) at this same point. factD does
  // not appear in the Mott branch at all, which is why only the second one divides by it.
  //
  // This line read `* fm * fm;` and carried a comment saying factD "is set only for particles
  // with a magnetic-moment correction; it is zero for the particles here". It is not zero for
  // any of them: SetTargetMass is called by SetupTarget for EVERY target
  // (G4WentzelOKandVIxSection.cc:208) and SampleScattering calls SetupTarget immediately before
  // this function. docs/RISK.md V47 has what the missing factor was worth.
  real_t grej;
  if (s.use_mott) {
    const real_t beta = data::mott_beta<real_t>(z, s.tkin, s.mass);
    grej = data::mott_ratio<real_t>(z, beta, sqrt(z1)) * fm * fm;
  } else {
    const real_t fact_d = sqrt(s.mom2) / wv_target_mass<real_t>(z);
    grej = (real_t(1) - z1 * s.fact_b
            + wv_fact_b1<real_t>() * real_t(z) * sqrt(z1 * s.fact_b) * (real_t(2) - z1))
           * fm * fm / (real_t(1) + z1 * fact_d);
  }
  if (s.mott_factor * rng.uniform() <= grej) {
    real_t cost = real_t(1) - z1;
    if (cost > real_t(1)) { cost = real_t(1); }
    if (cost < real_t(-1)) { cost = real_t(-1); }
    const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
    const real_t phi = units::twopi<real_t>() * rng.uniform();
    out = Vec3<real_t>{sint * cos(phi), sint * sin(phi), cost};
  }
  return out;
}

template <typename real_t>
struct WentzelScatterResult {
  Vec3<real_t> dir;
  Vec3<real_t> displacement;
};

/// Gamma deviate with shape 2, as G4RandGamma::shoot(2, 2) returns scaled by the caller.
/// For integer shape k the sum of k exponentials is exact, so no rejection is needed.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t gamma2(Rng& rng) {
  return -(log(rng.uniform()) + log(rng.uniform()));
}

/// The mixed multiple/single scattering sampler.
/// Transcribed from G4WentzelVIModel::SampleScattering.
///
/// Walks the step alternating between multiple-scattering sub-steps (at most two, each
/// sampling a deflection from an exponential in the reduced angle) and explicit single
/// Coulomb scatters drawn at exponentially distributed intervals.
template <typename real_t, typename Rng>
__host__ __device__ inline WentzelScatterResult<real_t> wv_sample_scattering(
    const data::Material<real_t>& m, const ParticleDef<real_t>& pd, ParticleType type,
    const WentzelMscState<real_t>& st, const WentzelElementXs<real_t>& els, real_t cut,
    real_t cos_theta_lim, const Vec3<real_t>& old_dir, bool lat_displacement, Rng& rng) {
  WentzelScatterResult<real_t> out{old_dir, Vec3<real_t>{real_t(0), real_t(0), real_t(0)}};
  if (st.t_path <= real_t(0)) { return out; }

  const real_t invlambda =
      (st.lambda_eff < real_t(1e29)) ? real_t(0.5) / st.lambda_eff : real_t(0);
  int n_msc_steps = 1;
  real_t x0 = st.t_path;
  real_t z0 = x0 * invlambda;
  const real_t prob2 = real_t(0);  // useSecondMoment is false by default

  if (!st.single_scattering_mode) {
    constexpr real_t zzmin = real_t(0.05);
    if (z0 > zzmin) {
      x0 *= real_t(0.5);
      z0 *= real_t(0.5);
      n_msc_steps = 2;
    }
    if (z0 > zzmin) { z0 += exp(real_t(-1) / z0); }
  }

  real_t x1 = real_t(2) * st.t_path;
  if (st.xtsec > real_t(0)) { x1 = -log(rng.uniform()) / st.xtsec; }
  if (st.single_scattering_mode && x1 > st.t_path) { return out; }

  Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};
  out.displacement = Vec3<real_t>{real_t(0), real_t(0), -st.z_path};
  const real_t mscfac = st.z_path / st.t_path;
  real_t x2 = x0;

  for (int guard = 0; guard < 10000; ++guard) {
    if (st.single_scattering_mode && x1 > x2) {
      out.displacement = out.displacement + (x2 * mscfac) * dir;
      break;
    }
    real_t step;
    bool single_scat;
    if (x1 <= x2) {
      step = x1;
      single_scat = true;
    } else {
      step = x2;
      single_scat = false;
    }
    out.displacement = out.displacement + (step * mscfac) * dir;

    if (single_scat) {
      int i = 0;
      if (els.n > 1 && st.xtsec > real_t(0)) {
        const real_t qsec = rng.uniform() * st.xtsec;
        for (; i < els.n; ++i) {
          if (els.cumulative[i] >= qsec) { break; }
        }
        if (i >= els.n) { i = els.n - 1; }
      }
      const int z = static_cast<int>(m.z[i] + real_t(0.5));
      const WentzelState<real_t> s =
          wentzel_setup(pd, type, st.pre_kin_energy, m.inv_a23, z, cut, cos_theta_lim);
      const Vec3<real_t> t = wv_sample_single(s, z, st.cos_theta_min, s.cos_tet_max_nuc,
                                              els.electron_fraction[i], rng);
      dir = normalize(rotate_uz(t, dir));
      x2 -= step;
      x1 = (st.xtsec > real_t(0)) ? -log(rng.uniform()) / st.xtsec : real_t(2) * st.t_path;
    } else {
      --n_msc_steps;
      x1 -= step;
      x2 = x0;
      const bool is_first = !(prob2 > real_t(0) && rng.uniform() < prob2);
      real_t z;
      for (int g2 = 0; g2 < 1000; ++g2) {
        z = is_first ? -log(rng.uniform()) : gamma2<real_t>(rng);
        z *= z0;
        if (z <= real_t(1)) { break; }
      }
      real_t cost = real_t(1) - real_t(2) * z;
      if (cost > real_t(1)) { cost = real_t(1); }
      if (cost < real_t(-1)) { cost = real_t(-1); }
      const real_t sint = sqrt((real_t(1) - cost) * (real_t(1) + cost));
      const real_t phi = units::twopi<real_t>() * rng.uniform();
      const real_t vx1 = sint * cos(phi);
      const real_t vy1 = sint * sin(phi);

      if (lat_displacement) {
        constexpr real_t invsqrt12 = real_t(1) / real_t(3.4641016151377544);  // 1/sqrt(12)
        const real_t rms = invsqrt12 * sqrt(real_t(2) * z0);
        const real_t r = x0 * mscfac;
        const real_t dx = r * (real_t(0.5) * vx1 + rms * urban_gauss(real_t(0), real_t(1), rng));
        const real_t dy = r * (real_t(0.5) * vy1 + rms * urban_gauss(real_t(0), real_t(1), rng));
        const real_t d = r * r - dx * dx - dy * dy;
        if (d >= real_t(0)) {
          const Vec3<real_t> t{dx, dy, sqrt(d) - r};
          out.displacement = out.displacement + rotate_uz(t, dir);
        }
      }
      dir = normalize(rotate_uz(Vec3<real_t>{vx1, vy1, cost}, dir));
    }
    if (n_msc_steps <= 0) { break; }
  }

  out.dir = normalize(rotate_uz(dir, old_dir));
  out.displacement = rotate_uz(out.displacement, old_dir);
  return out;
}

}  // namespace g4gpu::em
