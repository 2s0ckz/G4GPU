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
//
// That last sentence is now history rather than a plan: P8 did it, and the paragraph below says
// what it found on the way.
// ---------------------------------------------------------------------------------------------
// P8: THE TWO HALVES ARE ONE FILE'S WORTH OF PHYSICS AND WERE TWO TRANSCRIPTIONS OF IT
//
// This header and `xs/neutron_general_xs.cuh` were written by different packages against the
// same Geant4 class. P1 wrote the socket - the grid constants, the contract, and a
// `log_vector_value` so that `step_neutral` could look a row up; P2 wrote the table - the same
// grid, built as `G4PhysicsLogVector`'s constructor builds it, and `phys_vec_log_value`, and
// checked the result against the oracle bit for bit.
//
// They did not agree. Both compute a log vector's node energies, and P2's own header names the
// two wrong ways of doing it:
//
//     P1  x[j] = e_min * pow(r, j),  r = exp((log(e_max) - log(e_min)) / n_bins)
//     P2  x[j] = e_min * exp(j / invdBin),  invdBin = (n_nodes-1) / log(e_max/e_min),
//         with x[0] and x[n-1] assigned exactly - which is G4PhysicsLogVector::Initialise
//
// `log(a) - log(b)` and `log(a/b)` are different doubles, and `pow(r, j)` is not `exp(j*ln r)`,
// so the interior nodes differ in the last places - and an interpolation AMPLIFIES that,
// because `(e - x1)/(x2 - x1)` divides by a bin width. `tests/test_wiring.cu` section 5
// measures it on the real table - G4NeutronElasticXS + G4NeutronInelasticXS +
// G4NeutronCaptureXS out of G4PARTICLEXS4.0, summed onto both grids for water and lead, at
// every node and every bin midpoint, 1884 points:
//
//     low zone (400 bins over 4.301 decades)    worst 7.62e-14 relative
//     high zone (70 bins over 6.699 decades)    worst 2.02e-16 relative
//
// The factor of 400 between the two zones is the amplification and not noise: the low grid's
// bins are 23 times narrower in log(e), so the same ulp of node energy is a larger fraction of
// `x2 - x1`. 7.6e-14 is two orders of magnitude above an ulp of the value being interpolated,
// from an input that was correct to an ulp - which is docs/RISK.md V37's mechanism again, and
// the reason a "few ulps" note in a header is not a reason to keep the second transcription.
//
// The size is still small compared with anything a dose comparison resolves. That is not the
// point either: the point is that the transport was reading the transcription no oracle had
// seen, while the one that is bit-exact against Geant4 sat in the next directory.
// docs/RISK.md V5/V7 is a Bragg peak 0.3% out for the same reason at a larger scale, and V40 is
// two answers to one question about a level table.
//
// So `NeutronGeneralXs` now holds P2's `PhysVec` views and evaluates them with P2's
// `phys_vec_log_value`. There is one grid, one lookup and one owner. What stays here is what is
// about the PROCESS rather than about the table: the tracking cut, the sub-process order, and
// the contract that a table may not arrive without its final states.
#pragma once
#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/xs/neutron_general_xs.cuh"

namespace g4gpu::had {

/// P2's cross sections live in `g4gpu::hadronic::xs`, which unqualified lookup cannot reach
/// from inside `g4gpu::had` without help on every mention. Named once, as
/// `capture/capture_process.cuh` does.
namespace xs = g4gpu::hadronic::xs;

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

/// The combined table, per material - as a DEVICE VIEW of the five tables
/// `xs::ngp_build_table` builds.
///
/// Five arrays of `xs::PhysVec`, one entry per material, and not five arrays of doubles: a
/// `PhysVec` is what P2's builder already produces (`NeutronGeneralTable::p0 .. p4`), it carries
/// the `Initialise()` results so no logarithm is recomputed per lookup, and it makes the node
/// energies the ones the table was BUILT on rather than a formula that agrees with them to a few
/// ulps. See the P8 note at the top of this file for what that formula cost.
///
/// A null view is the state the engine is in today, and `total()` then returns zero.
///
/// THE CONTRACT, which is what this file is for:
///
///   * `t0[m]` is the MACROSCOPIC cross section in 1/mm - sum over elements of
///     n_atoms[i] * sigma_i - of elastic + inelastic + capture, on the low grid, for material m.
///     `t3[m]` is the same for elastic + inelastic only. G4NeutronGeneralProcess::
///     BuildPhysicsTable computes exactly these and stores them in tables[0] and tables[3].
///   * `t1[m]` is sigma_el/sigma_total and `t2[m]` is (sigma_el+sigma_inel)/sigma_total -
///     CUMULATIVE, in that order, as tables[1] and tables[2]. `t4[m]` is sigma_inel/sigma_total,
///     as tables[4]. Note the ORDER SWAP between the zones: PostStepDoIt tests elastic first
///     below the middle energy and inelastic first above it, so the high zone's single partial
///     is the INELASTIC one. Getting that backwards exchanges two cross sections that differ by
///     a factor of a few and would still look like a plausible neutron.
///   * `n_materials` must match the scene's material count. A lookup with `material` outside
///     it is a caller error, not a clamped answer.
///
/// What is NOT here, and must arrive with it: the final states. A table with no sampler is a
/// neutron that decides to interact and then cannot, so `TransportEngine::Upload` refuses that
/// combination rather than letting the device discover it - see the note in step_neutral.
template <typename real_t>
struct NeutronGeneralXs {
  const xs::PhysVec<real_t>* t0 = nullptr;  ///< low grid, elastic + inelastic + capture
  const xs::PhysVec<real_t>* t1 = nullptr;  ///< low grid, sigma_el / total
  const xs::PhysVec<real_t>* t2 = nullptr;  ///< low grid, (sigma_el + sigma_inel) / total
  const xs::PhysVec<real_t>* t3 = nullptr;  ///< high grid, elastic + inelastic
  const xs::PhysVec<real_t>* t4 = nullptr;  ///< high grid, sigma_inel / total
  int n_materials = 0;

  /// G4PhysicsVector::LogVectorValue on the node energies of a G4PhysicsLogVector recomputed
  /// from its two ends, which is what this socket did before P8 unified it with P2's table.
  ///
  /// **NOT WHAT `total()` AND `select()` READ ANY MORE**, and kept for one reason:
  /// `tests/test_species.cu` (P1's) checks the lookup MECHANICS through it - that the bin is
  /// found from log(e) and the interpolation is then linear in e, which a row that is exactly
  /// linear in energy detects - and that claim is still true and still worth a test. What it
  /// cannot check is the node energies, because it generates its synthetic row from the same
  /// formula; `tests/test_wiring.cu` section 5 compares this function against
  /// `xs::phys_vec_log_value` on the grid P2's builder actually produces, which is where the
  /// disagreement is.
  ///
  /// Do not call it from new code. `xs::phys_vec_log_value` is the transcription that has been
  /// compared with Geant4.
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

  /// Combined macroscopic cross section, 1/mm. G4NeutronGeneralProcess::CurrentCrossSection,
  /// which is `xs::ngp_lambda`.
  ///
  /// `loge` IS AN ARGUMENT AND NOT COMPUTED HERE, because Geant4 computes it once per step:
  /// `ComputeGeneralLambda` and `GetProbability` both read the same `fLogEnergy`. A socket that
  /// took the logarithm twice would still be right and would be two transcendentals per step in
  /// a kernel that is already spilling (docs/RISK.md V22).
  ///
  /// The zone test is `energy <= fMiddleEnergy`, so exactly 20 MeV reads the LOW table - the one
  /// that includes capture.
  __host__ __device__ real_t total(int material, real_t ekin, real_t loge) const {
    if (ekin <= kNeutronXsEMiddle<real_t>()) {
      if (t0 == nullptr) { return real_t(0); }
      return xs::phys_vec_log_value(t0[material], ekin, loge);
    }
    if (t3 == nullptr) { return real_t(0); }
    return xs::phys_vec_log_value(t3[material], ekin, loge);
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
  ///
  /// One `xs::ngp_select_subprocess`, with the enum this port names the answer by. The two are
  /// asserted equal for every material, every node and both zones in `tests/test_wiring.cu`.
  __host__ __device__ NeutronSubProcess select(int material, real_t ekin, real_t loge,
                                               real_t q) const {
    if (ekin <= kNeutronXsEMiddle<real_t>()) {
      if (q <= xs::phys_vec_log_value(t1[material], ekin, loge)) {
        return NeutronSubProcess::kElastic;
      }
      if (q <= xs::phys_vec_log_value(t2[material], ekin, loge)) {
        return NeutronSubProcess::kInelastic;
      }
      return NeutronSubProcess::kCapture;
    }
    return (q <= xs::phys_vec_log_value(t4[material], ekin, loge))
               ? NeutronSubProcess::kInelastic
               : NeutronSubProcess::kElastic;
  }
};

/// The socket built from P2's host-side table. Host-only: `NeutronGeneralTable` holds
/// `std::vector`s, and what this returns points INTO them - so the result is valid as long as
/// the table is, and an upload to a device has to copy the PhysVec arrays and repoint their
/// `e`/`v` members. Written here rather than at the future call site because the mapping
/// t0..t4 -> the five contract rows above is the thing that must not be guessed twice.
template <typename real_t>
__host__ inline NeutronGeneralXs<real_t> neutron_general_view(
    const xs::NeutronGeneralTable<real_t>& t) {
  NeutronGeneralXs<real_t> v;
  v.t0 = t.p0;
  v.t1 = t.p1;
  v.t2 = t.p2;
  v.t3 = t.p3;
  v.t4 = t.p4;
  v.n_materials = t.n_mat;
  return v;
}

}  // namespace g4gpu::had
