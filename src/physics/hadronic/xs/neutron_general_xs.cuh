// THE NEUTRON GRID: G4NeutronGeneralProcess's combined per-material table.
//
// Transcribed from source/processes/hadronic/processes/src/G4NeutronGeneralProcess.cc and its
// header's inlines (11.1.1): PreparePhysicsTable's grid construction, BuildPhysicsTable's five
// tables, ComputeCrossSection, CurrentCrossSection, ComputeGeneralLambda, GetProbability, and
// PostStepDoIt's sub-process selection.
//
// WHY THIS FILE EXISTS AT ALL
//
// `ref/oracle/hadronic_params.csv` says `EnableNeutronGeneralProcess = 1` in 11.1.1, and
// G4HadronInelasticQBBC::ConstructProcess sets it explicitly (`param->SetEnableNeutronGeneral-
// Process(true)`). So a neutron in QBBC has ONE discrete process, not three: elastic,
// inelastic and capture are its sub-processes, it builds a combined cross-section table per
// material at initialisation, and the transport reads THAT table. The three data sets in
// particlexs.cuh are inputs to it and are not what a step length is computed from.
//
// This is docs/RISK.md V5/V7 applied before a number is compared. A port that got
// G4NeutronElasticXS exactly right and then evaluated it per step would be right about the
// cross section and wrong about the transport, by whatever the difference is between a model
// and a 401-point linear interpolation of it - and the disagreement would be blamed on the
// data set.
//
// THE GRID, EXACTLY
//
//   fMinEnergy    = 1 keV        fMiddleEnergy = 20 MeV
//   fMaxEnergy    = max(100 MeV, G4HadronicParameters::GetMaxEnergy()) = 100 TeV
//   nLowE  = 100 * G4lrint(log10(fMiddleEnergy/fMinEnergy)) = 100 * lrint(4.301) = 400 bins
//   nHighE =  10 * G4lrint(log10(fMaxEnergy/fMiddleEnergy)) =  10 * lrint(6.699) =  70 bins
//
//   aVector = G4PhysicsLogVector(1 keV,  20 MeV,  400, false)   401 nodes, tables 0, 1, 2
//   bVector = G4PhysicsLogVector(20 MeV, 100 TeV,  70, false)    71 nodes, tables 3, 4
//
// The two `G4lrint` calls are the load-bearing part. log10(20000) is 4.301, not 4, and
// log10(5e6) is 6.699, not 7: the rounding is to nearest, so the low grid gets 400 bins over
// 4.301 decades (93 per decade, not 100) and the high grid 70 over 6.699 (10.4 per decade).
// Reading the `100` and `10` as bins-per-decade and building 431 and 67 bins would put every
// interior node in a different place.
//
// A second subtlety: the tables are `new G4PhysicsVector(aVector)`, a copy into the BASE
// class. G4PhysicsVector's copy constructor is `= default`, so `type` comes across as
// T_G4PhysicsLogVector along with invdBin and logemin, and LogVectorValue works. If the copy
// had been a slice, `type` would have been the default T_G4PhysicsFreeVector and every lookup
// would have gone through std::lower_bound instead - same answer, since the nodes are the
// same, which is why this is worth a comment rather than a check.
//
// THE FIVE TABLES
//
//   0  low grid   sigEl + sigInel + sigCap    macroscopic, 1/mm - the total interaction length
//   1  low grid   sigEl / sum
//   2  low grid   (sigEl + sigInel) / sum
//   3  high grid  sigEl + sigInel             capture is exactly zero above 20 MeV
//   4  high grid  sigInel / sum
//
// and the sub-process is chosen from a uniform q: below 20 MeV, elastic if q <= table1,
// inelastic if q <= table2, capture otherwise; above 20 MeV, inelastic if q <= table4, else
// elastic. Note the *order* differs between the two zones - elastic first below, inelastic
// first above - which matters for reproducing a random stream, not for the table.
//
// `sigX` is `fXSFactor * sum_i natom_i * xs->ComputeCrossSectionPerElement(...)`, i.e. the
// ELEMENT cross section weighted by atom density. No isotope is selected while the table is
// built; SampleZandA happens later, inside the chosen sub-process. The two factors are 1
// unless G4HadronicParameters::ApplyFactorXS is on, and hadronic_params.csv says it is not.
//
// WHAT IS NOT A CROSS SECTION AND IS STILL IN HERE
//
// `fTimeLimit = 10 us`. G4NeutronTrackingCut::ConstructProcess returns immediately when a
// G4NeutronGeneralProcess exists, so QBBC's neutron time cut is this member and not a
// G4NeutronKiller. PostStepGetPhysicalInteractionLength returns 0 when the track's global
// time has reached it, and PostStepDoIt then kills the track. The killer's kinetic-energy cut
// is 0 MeV, i.e. nothing. Carried here as a constant because it belongs to the same object
// the table does, and P1/P8 need it where they find the table.
#pragma once
#include <cmath>
#include <vector>

#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/physics_vector.cuh"

namespace g4gpu::hadronic::xs {

template <typename real_t> __host__ __device__ constexpr real_t ngp_min_energy() {
  return real_t(1) * units::keV<real_t>();
}
template <typename real_t> __host__ __device__ constexpr real_t ngp_middle_energy() {
  return real_t(20) * units::MeV<real_t>();
}
/// max(100 MeV, G4HadronicParameters::GetMaxEnergy()), and GetMaxEnergy defaults to 100 TeV
/// (G4HadronicParameters.cc: `fMaxEnergy = 100.0*CLHEP::TeV`). ref/oracle/hadronic_params.csv
/// carries the installed value, and tests/test_hadronic_xs.cu checks this against it.
template <typename real_t> __host__ __device__ constexpr real_t ngp_max_energy() {
  return real_t(1e8) * units::MeV<real_t>();  // 100 TeV
}
/// fTimeLimit - the neutron time cut, which lives in this process and not in G4NeutronKiller.
template <typename real_t> __host__ __device__ constexpr real_t ngp_time_limit() {
  return real_t(10) * real_t(1000) * units::ns<real_t>();  // 10 microsecond
}

/// `nLowE` and `nHighE` after PreparePhysicsTable multiplies them, i.e. bin counts and not
/// bins per decade. G4lrint is round-half-to-even; std::lrint in the default rounding mode is
/// the same function.
__host__ __device__ inline int ngp_n_low_bins() {
  return 100 * static_cast<int>(lrint(log10(20.0 / 1.0e-3)));
}
__host__ __device__ inline int ngp_n_high_bins() {
  return 10 * static_cast<int>(lrint(log10(1.0e8 / 20.0)));
}

/// One G4PhysicsLogVector's node energies, built the way its constructor builds them:
/// the two ends are assigned exactly, Initialise() computes invdBin from them, and the
/// interior is `edgeMin*G4Exp(i/invdBin)`.
///
/// Not `edgeMin*pow(ratio, i/n)` and not `exp(log(emin) + i*d)`: those agree to a few ulps
/// and this one is what the nodes are.
template <typename real_t>
__host__ inline void ngp_build_log_grid(real_t emin, real_t emax, int nbin,
                                        std::vector<real_t>& e) {
  int n = nbin + 1;
  if (n < 3) { n = 3; }
  e.assign(static_cast<std::size_t>(n), real_t(0));
  e[0] = emin;
  e[static_cast<std::size_t>(n - 1)] = emax;
  const int idxmax = n - 2;
  const real_t inv_dbin = static_cast<real_t>(idxmax + 1) / log(emax / emin);
  for (int i = 1; i <= idxmax; ++i) {
    e[static_cast<std::size_t>(i)] = emin * exp(static_cast<real_t>(i) / inv_dbin);
  }
}

/// The five tables, for every material, plus the two grids they sit on.
template <typename real_t>
struct NeutronGeneralTable {
  int n_mat = 0;
  int n_low = 0;   ///< nodes, = nLowE + 1
  int n_high = 0;  ///< nodes, = nHighE + 1
  std::vector<real_t> e_low, e_high;
  /// Row-major, `[material*n_low + i]` and `[material*n_high + i]`.
  std::vector<real_t> t0, t1, t2, t3, t4;

  /// PhysVec views, filled after the tables are, so evaluation needs no logarithms.
  std::vector<PhysVec<real_t>> v0, v1, v2, v3, v4;
  /// The five pointers the evaluation reads. std::vector::operator[] is host-only and
  /// calling it from a __host__ __device__ function is an nvcc error; see the same note in
  /// data/particlexs_data.cuh.
  const PhysVec<real_t>* p0 = nullptr;
  const PhysVec<real_t>* p1 = nullptr;
  const PhysVec<real_t>* p2 = nullptr;
  const PhysVec<real_t>* p3 = nullptr;
  const PhysVec<real_t>* p4 = nullptr;
};

/// G4NeutronGeneralProcess::ComputeCrossSection - the element cross section of one data set,
/// weighted by atom density, summed over the material's elements.
///
/// `natom` is G4Material::GetVecNbOfAtomsPerVolume(), which data::Material carries as
/// `n_atoms[]` in atoms per mm^3, so the result is a macroscopic cross section in 1/mm.
template <typename real_t>
__host__ inline real_t ngp_material_xs(const PxsDataSet<real_t>& ds,
                                       const data::Material<real_t>& mat, real_t e,
                                       real_t loge) {
  real_t sig = real_t(0.0);
  for (int i = 0; i < mat.n_elements; ++i) {
    const int z = static_cast<int>(mat.z[i]);
    const XsValue<real_t> x = pxs_element_xs<real_t>(ds, e, loge, z);
    xs_fatal_if_refused(x.refused, "ngp_material_xs");
    sig += mat.n_atoms[i] * x.value;
  }
  return sig;
}

/// PreparePhysicsTable + BuildPhysicsTable, for a list of materials.
///
/// @param el, inel, cap  the three data sets G4NeutronGeneralProcess takes as
///                       `InitialisationXS(proc)` = the FIRST data set of each sub-process's
///                       store, which in QBBC are G4NeutronElasticXS, G4NeutronInelasticXS
///                       and G4NeutronCaptureXS.
/// @param xs_factor_el, xs_factor_inel  fXSFactorEl / fXSFactorInel. They are 1 unless
///                       G4HadronicParameters::ApplyFactorXS is set, which it is not in
///                       11.1.1 - passed rather than assumed so the assumption is visible.
template <typename real_t>
__host__ inline void ngp_build_table(const PxsDataSet<real_t>& el,
                                     const PxsDataSet<real_t>& inel,
                                     const PxsDataSet<real_t>& cap,
                                     const data::Material<real_t>* mats, int n_mat,
                                     NeutronGeneralTable<real_t>& t,
                                     real_t xs_factor_el = real_t(1),
                                     real_t xs_factor_inel = real_t(1)) {
  ngp_build_log_grid<real_t>(ngp_min_energy<real_t>(), ngp_middle_energy<real_t>(),
                             ngp_n_low_bins(), t.e_low);
  ngp_build_log_grid<real_t>(ngp_middle_energy<real_t>(), ngp_max_energy<real_t>(),
                             ngp_n_high_bins(), t.e_high);
  t.n_low = static_cast<int>(t.e_low.size());
  t.n_high = static_cast<int>(t.e_high.size());
  t.n_mat = n_mat;

  const std::size_t nl = static_cast<std::size_t>(n_mat) * t.e_low.size();
  const std::size_t nh = static_cast<std::size_t>(n_mat) * t.e_high.size();
  t.t0.assign(nl, real_t(0));
  t.t1.assign(nl, real_t(0));
  t.t2.assign(nl, real_t(0));
  t.t3.assign(nh, real_t(0));
  t.t4.assign(nh, real_t(0));

  for (int m = 0; m < n_mat; ++m) {
    // energy interval 0
    for (int j = 0; j < t.n_low; ++j) {
      const real_t e = t.e_low[static_cast<std::size_t>(j)];
      const real_t loge = log(e);
      const real_t sigEl = xs_factor_el * ngp_material_xs<real_t>(el, mats[m], e, loge);
      const real_t sigInel = xs_factor_inel * ngp_material_xs<real_t>(inel, mats[m], e, loge);
      const real_t sigCap = ngp_material_xs<real_t>(cap, mats[m], e, loge);
      const real_t sum = sigEl + sigInel + sigCap;
      const std::size_t k = static_cast<std::size_t>(m * t.n_low + j);
      t.t0[k] = sum;
      t.t1[k] = sigEl / sum;
      t.t2[k] = (sigEl + sigInel) / sum;
    }
    // energy interval 1
    for (int j = 0; j < t.n_high; ++j) {
      const real_t e = t.e_high[static_cast<std::size_t>(j)];
      const real_t loge = log(e);
      const real_t sigEl = xs_factor_el * ngp_material_xs<real_t>(el, mats[m], e, loge);
      const real_t sigInel = xs_factor_inel * ngp_material_xs<real_t>(inel, mats[m], e, loge);
      const real_t sum = sigEl + sigInel;
      const std::size_t k = static_cast<std::size_t>(m * t.n_high + j);
      t.t3[k] = sum;
      t.t4[k] = sigInel / sum;
    }
  }

  // The views. Every table on a grid shares that grid's nodes, so edge_min, edge_max,
  // inv_dbin and log_emin are the same for all three low tables and both high ones.
  auto make = [&](const std::vector<real_t>& vals, const std::vector<real_t>& grid, int n,
                  std::vector<PhysVec<real_t>>& out) {
    out.assign(static_cast<std::size_t>(n_mat), PhysVec<real_t>{});
    for (int m = 0; m < n_mat; ++m) {
      PhysVec<real_t>& pv = out[static_cast<std::size_t>(m)];
      pv.e = grid.data();
      pv.v = vals.data() + static_cast<std::size_t>(m * n);
      pv.n = n;
      pv.type = kLogVector;
      phys_vec_initialise(pv);
    }
  };
  make(t.t0, t.e_low, t.n_low, t.v0);
  make(t.t1, t.e_low, t.n_low, t.v1);
  make(t.t2, t.e_low, t.n_low, t.v2);
  make(t.t3, t.e_high, t.n_high, t.v3);
  make(t.t4, t.e_high, t.n_high, t.v4);
  t.p0 = t.v0.data();
  t.p1 = t.v1.data();
  t.p2 = t.v2.data();
  t.p3 = t.v3.data();
  t.p4 = t.v4.data();
}

/// G4NeutronGeneralProcess::CurrentCrossSection - the macroscopic cross section the transport
/// actually uses, 1/mm. `currentInteractionLength` is its reciprocal.
///
/// The zone test is `energy <= fMiddleEnergy`, so exactly 20 MeV reads the LOW table - which
/// is table 0, the one that includes capture. Above it capture is gone.
template <typename real_t>
__host__ __device__ inline real_t ngp_lambda(const NeutronGeneralTable<real_t>& t, int imat,
                                             real_t energy, real_t loge) {
  const PhysVec<real_t>& pv =
      (energy <= ngp_middle_energy<real_t>()) ? t.p0[imat] : t.p3[imat];
  return phys_vec_log_value(pv, energy, loge);
}

/// Which sub-process fired, from a uniform draw. 0 elastic, 1 inelastic, 2 capture.
///
/// PostStepDoIt's own order, which is not the same in the two zones.
template <typename real_t>
__host__ __device__ inline int ngp_select_subprocess(const NeutronGeneralTable<real_t>& t,
                                                     int imat, real_t energy, real_t loge,
                                                     real_t q) {
  if (energy <= ngp_middle_energy<real_t>()) {
    if (q <= phys_vec_log_value(t.p1[imat], energy, loge)) { return 0; }
    if (q <= phys_vec_log_value(t.p2[imat], energy, loge)) { return 1; }
    return 2;
  }
  if (q <= phys_vec_log_value(t.p4[imat], energy, loge)) { return 1; }
  return 0;
}

}  // namespace g4gpu::hadronic::xs
