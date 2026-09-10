// The neutron's combined cross-section table, and the socket the rest of it plugs into.
//
// WHY THIS SHAPE, AND WHY IT IS EMPTY
//
// `ref/oracle/hadronic_params.csv` says `EnableNeutronGeneralProcess = 1` in 11.1.1, and
// `ref/oracle/neutron_processes.csv` shows what that means in the physics list QBBC actually
// builds - the neutron has exactly three processes:
//
//     Transportation        type 1  subtype 91
//     Decay                 type 6  subtype 201
//     NeutronGeneralProc    type 4  subtype 116     (fNeutronGeneral)
//
// and NOT a hadElastic, a neutronInelastic, an nCapture or a nKiller. `G4HadProcesses::
// BuildNeutronElastic` and `BuildNeutronInelasticAndCapture` take the `useNeutronGeneral`
// branch and hand their processes to `G4NeutronGeneralProcess::Set{Elastic,Inelastic,Capture}
// Process` instead of registering them; `G4NeutronTrackingCut::ConstructProcess` finds the
// general process and `return`s before it can create a `G4NeutronKiller`.
//
// So a neutron in Geant4 11.1.1 has ONE discrete interaction length, drawn against a
// per-material table of elastic + inelastic + capture summed, with the sub-process chosen
// afterwards from cumulative partials on the same grid. That is the structure below. Building
// the port's neutron transport around three independent processes would have been a different
// competition with a different answer, and it is the shape of the reference that decides.
//
// **This header defines the table and does not fill it.** P2 produces the four cross sections
// (G4NeutronElasticXS, G4NeutronInelasticXS, G4NeutronCaptureXS, and the Glauber-Gribov
// hand-over above their data); P8 sums them onto the grid below, uploads the rows, and writes
// the final states. Until then `step_neutral` is handed a null pointer, its cross section is
// zero, and a neutron streams to the world boundary or dies on the time cut - which is exactly
// what a Geant4 neutron does with `NeutronGeneralProc` inactivated.
//
// WHOSE FILE THIS IS. `src/physics/hadronic/` belongs to packages P2 (`xs/`) and P5
// (`process.cuh`, `elastic/`); this one file is P1's, and it is here rather than under
// `src/core/` because what it describes is a hadronic process's table and not a property of a
// track. It is the socket and nothing else: no dataset is read here, no cross section is
// computed here, and the only executable code is the G4PhysicsVector lookup that the grid
// constants are meaningless without. P2's readers produce microscopic cross sections per
// element from `G4PARTICLEXSDATA` and do not overlap with it; when P8 fills these rows from
// them, the lookup should move to whatever shared G4PhysicsVector helper P2 has by then and
// this file keeps the grid, the contract and the sub-process order.
#pragma once
#include <cmath>

#include "core/units.cuh"

namespace g4gpu::had {

// ---------------------------------------------------------------- the grid
//
// Every number here is a member initialiser in G4NeutronGeneralProcess's constructor or a line
// of its PreparePhysicsTable, and none of them is a choice made here. Reproducing the grid is
// not tidiness: docs/RISK.md V5/V7 is a Bragg peak 0.3% out because the port integrated the
// same physics on a finer grid than the one Geant4 transports on. A cross section is a table.
//
//   G4NeutronGeneralProcess::G4NeutronGeneralProcess:
//     fMinEnergy(1*CLHEP::keV), fMiddleEnergy(20*CLHEP::MeV), fMaxEnergy(100*CLHEP::TeV),
//     fTimeLimit(10*CLHEP::microsecond)
//   PreparePhysicsTable:
//     fMaxEnergy = std::max(100*MeV, param->GetMaxEnergy());   // 100 TeV, see hadronic_params
//     nLowE  *= G4lrint(std::log10(fMiddleEnergy/fMinEnergy)); // 100 * lrint(4.301) = 400
//     nHighE *= G4lrint(std::log10(fMaxEnergy/fMiddleEnergy)); //  10 * lrint(6.699) =  70
//     G4PhysicsLogVector aVector(fMinEnergy, fMiddleEnergy, nLowE,  false);
//     G4PhysicsLogVector bVector(fMiddleEnergy, fMaxEnergy,  nHighE, false);
//
// `false` is the spline flag, so the interpolation is linear - unlike the dE/dx and range
// tables, which are splined. Two tables in the same transport with different interpolation
// rules is the sort of thing that is only visible if you go and read which flag was passed.

/// G4NeutronGeneralProcess::fMinEnergy, MeV. Below it every lookup returns the first node.
template <typename real_t> __host__ __device__ constexpr real_t kNeutronXsEMin() {
  return real_t(1e-3);
}
/// fMiddleEnergy, MeV: the boundary between the two zones. At or below it the table carries
/// capture; above it capture is gone and only elastic and inelastic compete.
template <typename real_t> __host__ __device__ constexpr real_t kNeutronXsEMiddle() {
  return real_t(20);
}
/// fMaxEnergy, MeV, after PreparePhysicsTable widens it to G4HadronicParameters::GetMaxEnergy.
template <typename real_t> __host__ __device__ constexpr real_t kNeutronXsEMax() {
  return real_t(1e8);
}
/// Bins - not nodes - in each zone. A G4PhysicsLogVector of n bins has n+1 nodes.
constexpr int kNeutronXsLowBins = 400;
constexpr int kNeutronXsHighBins = 70;
constexpr int kNeutronXsLowNodes = kNeutronXsLowBins + 1;
constexpr int kNeutronXsHighNodes = kNeutronXsHighBins + 1;

/// G4NeutronGeneralProcess::fTimeLimit, in this port's time unit (ns).
///
/// 10 microseconds. It is the `G4NeutronTrackingCut` default that the general process took
/// over: the constructor of the cut sets `timeLimit = 10*CLHEP::microsecond` and the general
/// process's own constructor independently sets `fTimeLimit(10*CLHEP::microsecond)`, so the
/// two agree and the cut applies whichever branch the physics list took.
template <typename real_t> __host__ __device__ constexpr real_t kNeutronTimeLimit() {
  return real_t(10000);  // ns
}

/// G4NeutronKiller::kinEnergyThreshold as G4NeutronTrackingCut sets it: `kineticEnergyLimit`
/// is 0.0 and the test is `GetKineticEnergy() < kinEnergyThreshold`, which no non-negative
/// energy satisfies. So the energy half of the "10 us, 0 MeV" tracking cut is INERT, and it is
/// inert twice over here - G4NeutronGeneralProcess has no energy cut at all, only the time one.
///
/// Written down rather than omitted because "0 MeV" reads like a threshold that does something,
/// and a reader who assumed it kills a neutron at rest would be wrong: a neutron that has run
/// out of energy in Geant4 keeps being transported until it captures or ages out.
template <typename real_t> __host__ __device__ constexpr real_t kNeutronEnergyLimit() {
  return real_t(0);
}

/// Which sub-process of the general process fired.
enum class NeutronSubProcess : int { kElastic = 0, kInelastic = 1, kCapture = 2 };

/// The combined table, per material.
///
/// Rows are pointers rather than fixed arrays, and that is deliberate: 471 doubles x 5 rows x
/// however many materials a scene has is P8's allocation to size and upload, and fixing it here
/// would put a `data::kMaxMaterials`-shaped block in every Scene whether a neutron can appear
/// or not. A null table is the state the engine is in today.
///
/// THE CONTRACT, which is what this file is for:
///
///   * `total_low[m * kNeutronXsLowNodes + j]` is the MACROSCOPIC cross section in 1/mm -
///     sum over elements of n_atoms[i] * sigma_i - of elastic + inelastic + capture, at node j
///     of the low zone for material m. `total_high` is the same for elastic + inelastic only.
///     G4NeutronGeneralProcess::BuildPhysicsTable computes exactly these and stores them in
///     tables[0] and tables[3].
///   * `p_elastic_low` is sigma_el/sigma_total and `p_el_inel_low` is
///     (sigma_el+sigma_inel)/sigma_total - CUMULATIVE, in that order, as tables[1] and
///     tables[2]. `p_inelastic_high` is sigma_inel/sigma_total, as tables[4]. Note the
///     ORDER SWAP between the zones: PostStepDoIt tests elastic first below the middle energy
///     and inelastic first above it, so the high zone's single partial is the INELASTIC one.
///     Getting that backwards exchanges two cross sections that differ by a factor of a few
///     and would still look like a plausible neutron.
///   * `n_materials` must match the scene's material count. A lookup with `material` outside
///     it is a caller error, not a clamped answer.
///
/// What is NOT here, and must arrive with it: the final states. A table with no sampler is a
/// neutron that decides to interact and then cannot, so `TransportEngine::Upload` refuses that
/// combination rather than letting the device discover it - see the note in step_neutral.
template <typename real_t>
struct NeutronGeneralXs {
  const real_t* total_low = nullptr;
  const real_t* p_elastic_low = nullptr;
  const real_t* p_el_inel_low = nullptr;
  const real_t* total_high = nullptr;
  const real_t* p_inelastic_high = nullptr;
  int n_materials = 0;

  /// G4PhysicsVector::LogVectorValue + ComputeLogVectorBin + Interpolation, with useSpline
  /// false, transcribed:
  ///
  ///     if (e > edgeMin && e < edgeMax) { idx = min(int((loge-logemin)*invdBin), idxmax);
  ///                                       res = y[idx] + (e-x[idx])/(x[idx+1]-x[idx])*dy; }
  ///     else if (e <= edgeMin) res = y[0];
  ///     else                   res = y[N-1];
  ///
  /// Two details that a rewrite gets wrong. The bin is found from log(e) and the interpolation
  /// is then LINEAR IN e, not in log(e) - mixing those up is a smooth, plausible, wrong curve.
  /// And `idxmax` is `numberOfNodes - 2`, so the last bin's index is clamped rather than the
  /// energy: an e a hair below edgeMax still interpolates inside the final bin.
  __host__ __device__ static real_t log_vector_value(const real_t* row, int n_nodes,
                                                     real_t e_min, real_t e_max, real_t e) {
    if (e <= e_min) { return row[0]; }
    if (e >= e_max) { return row[n_nodes - 1]; }
    const int n_bins = n_nodes - 1;
    const real_t log_min = log(e_min);
    const real_t inv_dbin = real_t(n_bins) / (log(e_max) - log_min);
    int idx = static_cast<int>((log(e) - log_min) * inv_dbin);
    if (idx > n_bins - 1) { idx = n_bins - 1; }
    if (idx < 0) { idx = 0; }
    // The node energies of a G4PhysicsLogVector: x[j] = e_min * (e_max/e_min)^(j/n_bins).
    const real_t r = exp((log(e_max) - log_min) / real_t(n_bins));
    const real_t x1 = e_min * pow(r, real_t(idx));
    const real_t x2 = x1 * r;
    const real_t y1 = row[idx];
    return y1 + (e - x1) / (x2 - x1) * (row[idx + 1] - y1);
  }

  /// Combined macroscopic cross section, 1/mm. Zero when the row is absent, which is the only
  /// state a caller can be in before P8 lands.
  __host__ __device__ real_t total(int material, real_t ekin) const {
    if (ekin <= kNeutronXsEMiddle<real_t>()) {
      if (total_low == nullptr) { return real_t(0); }
      return log_vector_value(total_low + material * kNeutronXsLowNodes, kNeutronXsLowNodes,
                              kNeutronXsEMin<real_t>(), kNeutronXsEMiddle<real_t>(), ekin);
    }
    if (total_high == nullptr) { return real_t(0); }
    return log_vector_value(total_high + material * kNeutronXsHighNodes, kNeutronXsHighNodes,
                            kNeutronXsEMiddle<real_t>(), kNeutronXsEMax<real_t>(), ekin);
  }

  /// Which sub-process fired, from one uniform random @p q in [0,1).
  ///
  /// G4NeutronGeneralProcess::PostStepDoIt, verbatim in structure:
  ///
  ///     if (0 == idxEnergy) {                       // at or below fMiddleEnergy
  ///       if      (q <= GetProbability(1)) elastic
  ///       else if (q <= GetProbability(2)) inelastic
  ///       else                             capture
  ///     } else {                                    // above it
  ///       if (q <= GetProbability(4)) inelastic
  ///       else                        elastic
  ///     }
  __host__ __device__ NeutronSubProcess select(int material, real_t ekin, real_t q) const {
    if (ekin <= kNeutronXsEMiddle<real_t>()) {
      const real_t p_el =
          log_vector_value(p_elastic_low + material * kNeutronXsLowNodes, kNeutronXsLowNodes,
                           kNeutronXsEMin<real_t>(), kNeutronXsEMiddle<real_t>(), ekin);
      if (q <= p_el) { return NeutronSubProcess::kElastic; }
      const real_t p_ei =
          log_vector_value(p_el_inel_low + material * kNeutronXsLowNodes, kNeutronXsLowNodes,
                           kNeutronXsEMin<real_t>(), kNeutronXsEMiddle<real_t>(), ekin);
      if (q <= p_ei) { return NeutronSubProcess::kInelastic; }
      return NeutronSubProcess::kCapture;
    }
    const real_t p_inel = log_vector_value(p_inelastic_high + material * kNeutronXsHighNodes,
                                           kNeutronXsHighNodes, kNeutronXsEMiddle<real_t>(),
                                           kNeutronXsEMax<real_t>(), ekin);
    return (q <= p_inel) ? NeutronSubProcess::kInelastic : NeutronSubProcess::kElastic;
  }
};

}  // namespace g4gpu::had
