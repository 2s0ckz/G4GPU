// P19: the photon's, electron's, positron's and muon's nuclear interactions, as the STEPPERS see
// them - which process each species carries, where it sits among that species' other
// processes, its cross section per unit volume, and the one uniform that decides it.
//
// The sibling of `inelastic_wiring.cuh`, with the same division of labour: this file owns no
// physics. The cross sections are P2's and P13's (`xs/particlexs.cuh`'s `G4GammaNuclearXS` arm,
// `xs/chips_electronuclear.cuh`, `xs/kokoulin_muon_xs.cuh`), the models are P13's
// (`emextra/photon_nuclear.cuh`, `emextra/lepton_nuclear.cuh`), and the framework is P5's and
// P15's. What P19 owns is the answer to "for THIS particle at THIS energy in THIS material, is
// there a nuclear interaction, how likely, and which kernel runs it" - and one of those answers
// is not where anybody expected it.
//
// ---------------------------------------------------------------------------------------------
// WHICH PROCESS, read off the constructed QBBC (`ref/oracle/emextra_config.csv`, P13)
//
//     gamma     photonNuclear    inside GammaGeneralProc, NOT on the manager   GammaNuclearXS
//     e-        electronNuclear  on the manager, last of seven                 ElectroNuclearXS
//     e+        positronNuclear  on the manager, last of eight                 ElectroNuclearXS
//     mu-, mu+  muonNuclear      on the manager, before Decay                  KokoulinMuonNuclearXS
//
// Each is a `G4HadronInelasticProcess`, so `G4HadronicProcess::PostStepDoIt` is what applies it
// and P15's interaction queue is how this port reaches it (`interaction_apply.cuh`'s
// `run_emextra`). The three lepton processes are ordinary discrete competitors of their
// particles; the photon's is not a competitor at all.
//
// ---------------------------------------------------------------------------------------------
// THE PHOTON'S IS A SLICE OF ONE UNIFORM, AND ABOVE 100 MeV THE SLICE BELONGS TO CONVERSION
//
// `G4EmStandardPhysics` makes `phot`, `compt`, `conv` and `Rayl` sub-processes of ONE
// `G4GammaGeneralProcess` and `G4EmExtraPhysics::ConstructGammaElectroNuclear` hands it
// `photonNuclear` through `AddHadProcess` (physics_lists/constructors/electromagnetic/src/
// G4GammaGeneralProcess.cc). The general process draws one interaction length from one summed
// table, and `PostStepDoIt` draws ONE uniform `q` and walks the sub-processes' shares:
//
//   zone 0, E < 150 keV       phot, Rayl, compt                       no photonNuclear
//   zone 1, E < 2 m_e         phot, compt, Rayl                       no photonNuclear
//   zone 2, E < 100 MeV       q <= P7 conv, q <= P8 compt, q <= P9 phot, else photonNuclear
//   zone 3, E >= 100 MeV      q + P11 <= 1 conv, q + P12 <= 1 compt, q + P13 <= 1 phot,
//                             theGammaNuclear && q + P14 <= 1 photonNuclear,
//                             else if (theConversionMM) mu-pair, else conv
//
// with `BuildPhysicsTable` filling, in zone 3,
//
//     table 13 = (sigN + sigM)/sum        table 14 = sigN/sum
//
// where `sigM` is `G4GammaConversionToMuons`' cross section - which QBBC does not build
// (`gmumuActivated` false, asserted absent by P13). With sigM exactly 0.0, `sigN + sigM` IS
// `sigN` to the last bit, tables 13 and 14 hold identical numbers and identical spline
// coefficients, and the photonNuclear branch tests the complement of the condition that just
// failed: it CANNOT be taken. `theConversionMM` is null, so the share falls to the final
// `SelectEmProcess(step, theConversionEE)`. **Above 100 MeV QBBC's photon never interacts with
// a nucleus**: its photo-nuclear cross section is in the total, so the photon interacts as
// often as it would, and that fraction of its interactions are pair conversions. This is a port
// of 11.1.1 and transcribes the branch as it runs, whatever a later release did with it; the
// running Geant4 was asked rather than read - `ref/gammagp/` counts which sub-process takes a
// photon's first interaction, and at 99.9 MeV in water photonNuclear takes its share where at
// 100 MeV it takes none of a million (docs/RISK.md V207).
//
// Three consequences, and each is a number the brief asked for:
//
//   * The only photo-nuclear model QBBC's photon reaches is `G4LowEGammaNuclearModel`
//     (GammaNPreco, 0 - 200 MeV). Bertini's photon arm starts at 199 MeV and the QGS generator
//     at 3 GeV, both above the 100 MeV edge, so their rates through the photon process are ZERO
//     - by construction, not by statistics. (The two LEPTON models reach Bertini through their
//     own `G4CascadeInterface` members, which is a different road.)
//   * The giant resonance, at 10 - 30 MeV, is inside zone 2 and fully reachable. A 100 MeV
//     primary is in zone 3 - `preStepKinEnergy < minMMEnergy` is false at exactly 100 - so in
//     the B1 sweep's 100 MeV beam it is the cascade's photons that react, never the primary.
//   * Below 2 m_e there is no photo-nuclear term at all, so a run whose photons never exceed
//     1.022 MeV is untouched by this package to the last bit. That is the bit-identity the P19
//     brief asks for and it is ASSERTED (tests/test_emextra_transport.cu), not argued.
//
// `GammaGeneralProcess::kOff` is `/process/em/UseGeneralProcess false`, where `photonNuclear`
// is an ordinary process on the gamma's manager with its own interaction length at EVERY
// energy. It is the configuration the like-for-like photon column used before P19 (because a
// sub-process of the general process cannot be inactivated) and it is kept as a study switch,
// because it is the one in which Bertini's photon arm and the QGS refusal become reachable.
//
// THE REST OF THE PHOTON'S STEP, AND WHAT P21 CHANGED IN IT. The port's photon is P1's: four
// cross sections evaluated from the models every step, one length uniform and one selection
// uniform walked as Compton, conversion, photoelectric, Rayleigh. The general process differed
// from that in two more ways than photonNuclear, P19 measured both and changed neither, and P21
// transcribed both:
//
//   * RAYLEIGH IS ABSENT FROM 2 m_e UP (zones 2 and 3 sum no `sigR`): 0 Rayleigh scatters counted
//     in forty million first interactions at 1.5 MeV in water and in bone, where the cross
//     section's share is 17,293 and 30,846 of them (docs/RISK.md V208, V223).
//     `em::gamma_macroscopic_xs` takes the general process's state and leaves the term out, so
//     the total, the length and the walk are the zones'.
//   * IN ZONE 2 THE PHOTO-NUCLEAR SHARE IS TABLE 9, not sigN/sum: `BuildPhysicsTable` fills
//     `(sigConv + sigComp + sigPE)/sum` - 1.0 where sigN is 0 - at the 51 nodes of
//     `G4PhysicsLogVector(minEEEnergy, minMMEnergy, 50, false)`, 9.6% apart, NO spline, and
//     `PostStepDoIt` reads it through `LogVectorValue`, linear in E between two nodes. Across
//     the giant resonance that is 16% under sigN/sum at 22 MeV in bone and 8% over at 20 MeV in
//     water, and the running Geant4 follows the table (V209). `GammaGeneralTable9` below is that
//     table, built at upload from this port's cross sections at Geant4's own nodes, and
//     `step_gamma` selects photonNuclear in zone 2 exactly when `q > P9(E)` - the same
//     comparison on the same uniform as `case 2` of `PostStepDoIt`.
//
// What zone 2's TOTAL is, then. Geant4's is table 6, the linear interpolant of the node sums;
// this port keeps P1's direct EM total `xs.total` - it does not read tables 6, 7 and 8, whose
// difference from the models is EM interpolation, measured and recorded in docs/RISK.md - and
// makes the photo-nuclear part of its total the one table 9 implies: `xs.total / P9`, so that
// the share of the photon's interactions that go to a nucleus is `1 - P9` exactly and the EM
// rate is still the models'. The per-step store evaluation `sigN` needed is gone from zone 2.
//
// AND THE TABLE HAS AN EDGE THE CROSS SECTION DOES NOT. Between a material's last node where sigN
// is 0 and its threshold (11.08 - 11.50 MeV in water, 5.32 - 5.5 in compact bone, 3.69 - 4.0 in
// A-150) P9 interpolates below 1 where the data store's cross section is exactly 0, so Geant4
// selects photonNuclear on a zero cross section - 126 times in a million first interactions at
// 11.2 MeV in water, counted - and `SampleZandA` walks all-zero partial sums: one uniform, the
// material's FIRST element (`interaction_apply.cuh`'s `run_emextra` does the same, and counts it,
// `kEmxTable9Edge`).
//
// The photo-nuclear slice is placed where the general process places it RELATIVE TO THE ONE
// UNIFORM: it is the TOP slice of `q` in zone 2 in both codes (`q > P9` in both, since P21), so
// for a given `q` the two agree on whether the photon reacts with a nucleus - which is the sense
// in which the selection is on Geant4's stream.
//
// ---------------------------------------------------------------------------------------------
// THE THREE LEPTON PROCESSES ARE fHadNoIntegral, AND THE TARGET IS DRAWN AT THE PRE-STEP ENERGY
//
// `G4HadronicProcess::BuildPhysicsTable` (management/src/G4HadronicProcess.cc:194-199):
//
//     G4bool isLepton = (p.GetLeptonNumber() != 0);
//     if(charge != 0.0 && useIntegralXS && !isLepton && ok) { ... fXSType = ... }
//
// so for e+-, mu+- the integral approach is OFF: no rejection uniform, no recompute at the end
// of the step. `PostStepGetPhysicalInteractionLength` computes the cross section at the
// PRE-step energy (`DefineXSandMFP`), and `SampleZandA` in `PostStepDoIt` walks the partial
// sums that call left in the data store - so the element is drawn from the pre-step energy's
// partials, while the model is handed the post-step energy (`thePro.Initialise(aTrack)`). The
// queue carries `ekin_pre`, and `run_emextra` rebuilds the partials there. The photon is
// neutral and has no along-step loss, so for it the two energies are one.
//
// WHAT A MUON DOES THAT NOTHING ELSE HERE DOES. `G4KokoulinMuonNuclearXS` is a 61-node log
// vector from 1 GeV to 1 PeV and `G4PhysicsVector::Value` clamps BELOW its first node as well
// as above its last (docs/RISK.md V178), so a muon of any energy has the 1 GeV cross section -
// about 7.7e-8 /mm in water, a 13 km mean free path - and draws a length every step. The
// model then returns the muon unchanged below T = 563.5 MeV (`epmax <= CutFixed`). Both are
// Geant4's and both are transcribed; the first is why this package moves the random stream of
// every muon a proton beam makes, by one uniform a step, without the muon ever reacting.
//
// ---------------------------------------------------------------------------------------------
// THE THRESHOLDS ARE EXACT, AND THEY ARE WHAT KEEPS THIS OFF THE GAMMA GATE'S CLOCK
//
// Below a material's threshold every element's cross section is exactly zero - not small - and
// the steppers skip the evaluation there. For `G4GammaNuclearXS` it is the last leading-zero
// node of each element's vector (G4PhysicsVector::Value interpolates zero between two zeros and
// clamps to the first value below the first node); for `G4ElectroNuclearCrossSection` it is
// `max(EMi, ThresholdEnergy(Z, N))`, below which `GetElementCrossSection` returns 0 before
// touching a table. `host/hadronic_upload.cuh` computes both per material from the same
// functions the device calls, and `tests/test_emextra_wiring.cu` asserts, for every element of
// every material, that the cross section is exactly 0 AT the threshold and positive just above
// it - which is what makes skipping the evaluation an identity rather than an approximation.
// Water's photo-nuclear threshold is oxygen's 11.499 MeV and its electro-nuclear one oxygen's
// 7.296 MeV, so a 6 MeV photon in water, and every electron it makes, never evaluates either.
#pragma once

#include <cmath>
#include <type_traits>
#include <utility>

#include "core/particle.cuh"
#include "core/step_report.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/em/gamma_processes.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/chips_electronuclear.cuh"
#include "physics/hadronic/xs/kokoulin_muon_xs.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

namespace g4gpu::physics::hadronic::emextra {
// The muon model's 2.3 MB sampling table. Only a POINTER to it travels with the wiring, and the
// type is completed in `emextra/lepton_vd.cuh` - which carries Bertini and must not reach a
// stepping kernel (docs/RISK.md V188). A pointer to an incomplete type is what lets the
// stepper's header name it without including it.
struct MuVdTable;
}  // namespace g4gpu::physics::hadronic::emextra

namespace g4gpu::had {

namespace hxs = g4gpu::hadronic::xs;

// =============================================================================================
// Which process
// =============================================================================================

/// The four processes `G4EmExtraPhysics::ConstructProcess` builds in QBBC. Named as Geant4 names
/// them, which is what `emextra_config.csv` carries and `emextra::process_name` answers.
enum class EmExtraProcess : int {
  kNone = 0,
  kPhotonNuclear,    ///< "photonNuclear",   gamma, inside G4GammaGeneralProcess
  kElectronNuclear,  ///< "electronNuclear", e-
  kPositronNuclear,  ///< "positronNuclear", e+
  kMuonNuclear,      ///< "muonNuclear",     mu- and mu+ (one process object for both)
};

__host__ __device__ inline EmExtraProcess emextra_process_of(ParticleType t) {
  switch (t) {
    case ParticleType::kGamma:     return EmExtraProcess::kPhotonNuclear;
    case ParticleType::kElectron:  return EmExtraProcess::kElectronNuclear;
    case ParticleType::kPositron:  return EmExtraProcess::kPositronNuclear;
    case ParticleType::kMuonMinus:
    case ParticleType::kMuonPlus:  return EmExtraProcess::kMuonNuclear;
    default:                       return EmExtraProcess::kNone;
  }
}

__host__ __device__ inline const char* emextra_process_name(EmExtraProcess p) {
  switch (p) {
    case EmExtraProcess::kPhotonNuclear:   return "photonNuclear";
    case EmExtraProcess::kElectronNuclear: return "electronNuclear";
    case EmExtraProcess::kPositronNuclear: return "positronNuclear";
    case EmExtraProcess::kMuonNuclear:     return "muonNuclear";
    case EmExtraProcess::kNone:            break;
  }
  return "(none)";
}

/// What a step reports as the process that defined it. `core/step_report.cuh` has carried the
/// three names since P1, waiting for this package; `fHadronInelastic` would have been true of
/// the process CLASS and wrong about which process it was.
__host__ __device__ inline ProcessId emextra_process_id(EmExtraProcess p) {
  switch (p) {
    case EmExtraProcess::kPhotonNuclear:   return ProcessId::fPhotoNuclear;
    case EmExtraProcess::kElectronNuclear:
    case EmExtraProcess::kPositronNuclear: return ProcessId::fElectroNuclear;
    case EmExtraProcess::kMuonNuclear:     return ProcessId::fMuonNuclear;
    case EmExtraProcess::kNone:            break;
  }
  return ProcessId::fNotDefined;
}

// =============================================================================================
// The gamma general process and where photonNuclear sits in it
// =============================================================================================

/// `/process/em/UseGeneralProcess`. ON is QBBC as it ships; OFF is the configuration the
/// like-for-like photon column had to use while `photonNuclear` was absent from this port.
enum class GammaGeneralProcess : int { kOff = 0, kOn = 1 };

/// `minPEEnergy`, `minEEEnergy` and `minMMEnergy` from `G4GammaGeneralProcess`'s constructor:
/// `150*CLHEP::keV`, `2*CLHEP::electron_mass_c2`, `100*CLHEP::MeV`. The zone edges. The middle
/// one is `em::gamma_general_min_ee`'s, where `gamma_macroscopic_xs` drops Rayleigh, so that
/// the two edges cannot be two numbers.
template <typename real_t>
__host__ __device__ inline constexpr real_t gamma_general_min_pe() { return real_t(0.150); }
template <typename real_t>
__host__ __device__ inline constexpr real_t gamma_general_min_ee() {
  return em::gamma_general_min_ee<real_t>();
}
template <typename real_t>
__host__ __device__ inline constexpr real_t gamma_general_min_mm() { return real_t(100); }

/// `G4GammaGeneralProcess::TotalCrossSectionPerVolume`'s zone, which is also `idxEnergy` in
/// `PostStepDoIt`. Strict `<` on every edge, as the source has it, so exactly 100 MeV is zone 3.
template <typename real_t>
__host__ __device__ inline int gamma_general_zone(real_t e) {
  if (e < gamma_general_min_pe<real_t>()) { return 0; }
  if (e < gamma_general_min_ee<real_t>()) { return 1; }
  if (e < gamma_general_min_mm<real_t>()) { return 2; }
  return 3;
}

/// What the top slice of the selection uniform - the one `photonNuclear`'s share occupies -
/// does, at this energy, in this configuration.
enum class GammaNuclearSlot : int {
  /// No photo-nuclear term in the total at all: zones 0 and 1 of the general process.
  kAbsent = 0,
  /// The slice selects `photonNuclear`: zone 2 of the general process, or any energy with the
  /// general process off.
  kPhotoNuclear,
  /// The slice selects `conv`: zone 3 of the general process in 11.1.1, where table 14 equals
  /// table 13 because QBBC builds no `G4GammaConversionToMuons`. See this file's header.
  kConversion,
};

template <typename real_t>
__host__ __device__ inline GammaNuclearSlot gamma_nuclear_slot(GammaGeneralProcess mode,
                                                               real_t e) {
  if (mode == GammaGeneralProcess::kOff) { return GammaNuclearSlot::kPhotoNuclear; }
  const int zone = gamma_general_zone<real_t>(e);
  if (zone <= 1) { return GammaNuclearSlot::kAbsent; }
  if (zone == 2) { return GammaNuclearSlot::kPhotoNuclear; }
  return GammaNuclearSlot::kConversion;
}

/// `em::select_gamma_process` with the product `rand01 * total` handed in rather than formed
/// inside.
///
/// A SECOND COPY OF P1's WALK, AND BOUNDED BY A TEST. The photo-nuclear term makes the total the
/// selection uniform multiplies a different number from `xs.total`, so the EM walk has to be
/// entered with `q * (xs.total + sigN)` rather than `q * xs.total`. `step_gamma` calls P1's
/// function unchanged whenever the photo-nuclear term is zero - which is what makes those runs
/// bit-identical by construction rather than by care - and this one only when it is not.
/// `tests/test_emextra_wiring.cu` asserts the two agree on every `q` of a grid wherever they
/// are handed the same product.
template <typename real_t>
__host__ __device__ inline em::GammaProcess select_gamma_process_at(const em::GammaXS<real_t>& xs,
                                                                   real_t r) {
  if (xs.total <= real_t(0)) { return em::GammaProcess::kNone; }
  if (r < xs.compton) { return em::GammaProcess::kCompton; }
  r -= xs.compton;
  if (r < xs.pair) { return em::GammaProcess::kPair; }
  r -= xs.pair;
  if (r < xs.photoelectric) { return em::GammaProcess::kPhotoelectric; }
  return em::gamma_walk_last_slice(xs);
}

// =============================================================================================
// The device tables
// =============================================================================================

/// `nHighE`. Zone 2's vectors are `G4PhysicsLogVector(minEEEnergy, minMMEnergy, nHighE, false)`
/// (`G4GammaGeneralProcess::InitialiseProcess`, the `cVector` every table 6-9 is copied from):
/// 50 bins, 51 nodes, 9.6% apart, no spline.
inline constexpr int kGammaGeneralZone2Bins = 50;
inline constexpr int kGammaGeneralZone2Nodes = kGammaGeneralZone2Bins + 1;

/// P21: `G4GammaGeneralProcess`'s table 9 for every material of the scene - the fraction of a
/// zone-2 photon's interactions that are NOT photo-nuclear, `(sigConv + sigComp + sigPE)/sum`
/// at each node and 1.0 where sigN is 0 (`BuildPhysicsTable`, G4GammaGeneralProcess.cc:366-393).
/// Built once at upload by `host::build_emextra_host_tables` (`gamma_general_zone2_grid` and
/// `gamma_general_table9_node` below) from this port's cross sections - the numbers Geant4 builds
/// it from, to the EM lambda tables' interpolation (`tests/test_emextra_wiring.cu` section 8
/// holds it against the running Geant4's own table) - and read by `gamma_general_p9`, which is
/// `LogVectorValue`.
template <typename real_t>
struct GammaGeneralTable9 {
  const real_t* e = nullptr;  ///< the 51 node energies, MeV: `cVector`'s binVector
  const real_t* v = nullptr;  ///< `n_materials` x 51 values, row-major by material
  /// Per material: table 9 is EXACTLY 1.0 at and below this energy - the last node of its
  /// leading run of 1.0s, because `y1 + b*dy` with y1 = 1 and dy = 0 is 1.0 to the bit - so
  /// `step_gamma` evaluates nothing there and its step is the step with no photo-nuclear term.
  /// 0 when node 0 is already below 1: in air, argon's threshold is under 2 m_e.
  const real_t* threshold = nullptr;
  int n_materials = 0;
  real_t edge_min = 0;  ///< `edgeMin` = minEEEnergy
  real_t edge_max = 0;  ///< `edgeMax` = minMMEnergy
  real_t inv_dbin = 0;  ///< `invdBin` = (idxmax + 1)/G4Log(edgeMax/edgeMin)
  real_t log_emin = 0;  ///< `logemin` = G4Log(edgeMin)
};

/// Which threshold row `EmExtraTables::threshold` holds. The muon has none: its cross section is
/// the clamped Kokoulin table and is positive at every energy.
enum EmExtraThreshold : int {
  kGammaNuclearThreshold = 0,
  kElectroNuclearThreshold = 1,
  kNumEmExtraThresholds = 2,
};

/// Everything the four processes read that is not compiled in, as device pointers.
///
/// The two CHIPS parameterisations are function-scope constant tables and need no upload;
/// `G4GammaNuclearXS` below its 130 MeV data ceiling reads G4PARTICLEXS4.0/gamma, the muon's
/// cross section is `G4KokoulinMuonNuclearXS::BuildCrossSectionTable`'s 61-node vectors, and
/// the muon MODEL's 5 x 73 x 800 sampling table is `MakeSamplingTable`'s - all three built once
/// on the host by `host/hadronic_upload.cuh`, as Geant4 builds them once at initialisation.
///
/// Null pointers are the "no process" state: a run whose G4PARTICLEXSDATA could not be
/// resolved has no photo-nuclear cross section, which is a photon with no nuclear process, not
/// a photon whose nuclear process fails.
template <typename real_t>
struct EmExtraTables {
  const hxs::PxsDataSet<real_t>* gamma = nullptr;
  const hxs::kokoulin::KokoulinTable<real_t>* kokoulin = nullptr;
  const physics::hadronic::emextra::MuVdTable* mu_vd = nullptr;
  /// `kNumEmExtraThresholds * n_materials` energies, MeV, row-major by threshold kind: at or
  /// below `threshold[k * n_materials + m]` every element of material `m` has exactly zero
  /// cross section for process kind `k`. See the file header for how each is derived.
  const real_t* threshold = nullptr;
  int n_materials = 0;
  /// `kNumEmExtraThresholds * kEmExtraMaxZ` energies, MeV: the same two thresholds per ELEMENT,
  /// which `EmExtraXsFn::element` tests before it evaluates anything. The material's threshold
  /// is the smallest of its elements', so above it the others can still be exactly zero - in
  /// air at 6 MeV argon is open and carbon, nitrogen and oxygen are not - and the store would
  /// otherwise evaluate all four vectors at every photon step to add three zeros (docs/RISK.md
  /// V210 has what that cost the gamma gate). Zero for an element means "evaluate".
  const real_t* element_threshold = nullptr;
  /// P21: the general process's zone-2 table 9, one device struct. Null is "no photo-nuclear
  /// term in zone 2" - a run with no G4PARTICLEXS gamma data - exactly as a null `gamma` is.
  /// A pointer and not the struct, because this one travels by value in every kernel's wiring
  /// and the hadron kernels copy the wiring into their frames (docs/RISK.md V210): 8 bytes
  /// rather than 64.
  const GammaGeneralTable9<real_t>* p9 = nullptr;
};

/// The Z range `EmExtraTables::element_threshold` covers: every Z either CHIPS class answers
/// (`IsElementApplicable` is 0 < Z < 120 for the electro-nuclear one).
inline constexpr int kEmExtraMaxZ = 120;

/// The threshold for one (kind, material), or zero when none was uploaded - zero meaning
/// "evaluate at every energy", which is the answer that cannot be wrong.
template <typename real_t>
__host__ __device__ inline real_t emextra_threshold(const EmExtraTables<real_t>& t, int kind,
                                                    int mat) {
  if (t.threshold == nullptr || mat < 0 || mat >= t.n_materials) { return real_t(0); }
  return t.threshold[kind * t.n_materials + mat];
}

/// The threshold for one (kind, element), or zero - "evaluate" - when none was uploaded or Z is
/// outside the table.
template <typename real_t>
__host__ __device__ inline real_t emextra_element_threshold(const EmExtraTables<real_t>& t,
                                                            int kind, int Z) {
  if (t.element_threshold == nullptr || Z <= 0 || Z >= kEmExtraMaxZ) { return real_t(0); }
  return t.element_threshold[kind * kEmExtraMaxZ + Z];
}

// =============================================================================================
// The cross sections
// =============================================================================================

/// The data set of one process at one energy, as the three functions `xs/sample_za.cuh`'s
/// `G4CrossSectionDataStore` asks for - the shape `InelasticXsFn` has for P15's five.
///
///   * `G4GammaNuclearXS` is P2's `PxsDataSet` of kind `kGammaNuclear`, and its own
///     `SelectIsotope` (abundance times `IsoCrossSection`, or abundance alone above 150 MeV) is
///     `store_abundance_only`'s gamma branch.
///   * `G4ElectroNuclearCrossSection` and `G4KokoulinMuonNuclearXS` have `IsElementApplicable`
///     true (for 0 < Z < 120, and always) and override neither `GetIsoCrossSection` nor
///     `SelectIsotope`, so the store takes the element path and the isotope is drawn from the
///     abundances alone by `G4VCrossSectionDataSet::SelectIsotope`. `isotope()` is therefore
///     never reached for them, and answers with the element value rather than a zero that could
///     be mistaken for a cross section.
///
/// `eln` is `G4ElectroNuclearCrossSection`'s own per-Z cache (`lastE`, `lastG`, `lastSig`, the
/// J-function rows), which `GetElementCrossSection` rebuilds whenever Z changes. It is mutable
/// because the class is: the store calls a const functor and the real class updates members
/// under a const-looking interface.
template <typename real_t>
struct EmExtraXsFn {
  const EmExtraTables<real_t>* tables = nullptr;
  EmExtraProcess process = EmExtraProcess::kNone;
  real_t ekin = 0;
  real_t loge = 0;
  mutable hxs::chips::ElnState eln{};

  __host__ __device__ hxs::XsValue<real_t> element(int Z) const {
    // At or below the ELEMENT's threshold its cross section is exactly zero - the same identity
    // the material thresholds rest on, element by element - so nothing is evaluated. For the
    // electro-nuclear class this also leaves `eln`, its per-Z cache, as it was, which is what
    // `GetElementCrossSection` itself does below `EMi`; above `EMi` and at or below the nucleus'
    // `TH` Geant4 updates the cache and returns zero, and the cache's only reader is the next
    // call of this same function, which rebuilds it for its own Z.
    switch (process) {
      case EmExtraProcess::kPhotonNuclear:
        if (tables->gamma == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        if (!(ekin > emextra_element_threshold<real_t>(*tables, kGammaNuclearThreshold, Z))) {
          return {real_t(0), hxs::XsRefusal::kNone};
        }
        return hxs::pxs_element_xs<real_t>(*tables->gamma, ekin, loge, Z);
      case EmExtraProcess::kElectronNuclear:
      case EmExtraProcess::kPositronNuclear:
        if (!(ekin > emextra_element_threshold<real_t>(*tables, kElectroNuclearThreshold, Z))) {
          return {real_t(0), hxs::XsRefusal::kNone};
        }
        return hxs::chips::eln_element_xs<real_t>(eln, ekin, Z);
      case EmExtraProcess::kMuonNuclear:
        if (tables->kokoulin == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::kokoulin::element_xs<real_t>(*tables->kokoulin, ekin, Z);
      case EmExtraProcess::kNone:
        break;
    }
    return {real_t(0), hxs::XsRefusal::kNone};
  }

  __host__ __device__ hxs::XsValue<real_t> isotope(int Z, int A) const {
    if (process == EmExtraProcess::kPhotonNuclear && tables->gamma != nullptr) {
      return hxs::pxs_iso_xs<real_t>(*tables->gamma, ekin, loge, Z, A);
    }
    return element(Z);
  }

  __host__ __device__ bool abundance_only(int Z) const {
    if (process == EmExtraProcess::kPhotonNuclear && tables->gamma != nullptr) {
      return hxs::store_abundance_only<real_t>(*tables->gamma, Z, ekin);
    }
    return true;
  }
};

template <typename real_t>
__host__ __device__ inline EmExtraXsFn<real_t> emextra_xs_fn(const EmExtraTables<real_t>& t,
                                                            EmExtraProcess p, real_t ekin) {
  EmExtraXsFn<real_t> fn;
  fn.tables = &t;
  fn.process = p;
  fn.ekin = ekin;
  fn.loge = (ekin > real_t(0)) ? log(ekin) : real_t(0);
  return fn;
}

/// `G4CrossSectionDataStore::ComputeCrossSection` for one of the four processes, 1/mm, with the
/// partial sums `SampleZandA` reads. Exactly zero at or below the material's threshold, without
/// evaluating anything - see the file header for why that is an identity.
///
/// `__noinline__` FOR THE REASON `inelastic_xs_per_volume` GIVES AND ONE MORE: inlined into
/// `step_lepton` it would pull the CHIPS electro-nuclear J-function tables and their sampler's
/// arithmetic into a kernel whose frame is measured to the byte (docs/RISK.md V55, V63), and a
/// call boundary is what keeps the stepping kernel's own frame the one the gate was measured
/// with.
template <typename real_t>
__host__ __device__ __noinline__ real_t emextra_xs_per_volume(const EmExtraTables<real_t>& t,
                                                             EmExtraProcess p, int mat_index,
                                                             const data::Material<real_t>& mat,
                                                             real_t ekin,
                                                             hxs::MaterialXs<real_t>& mxs) {
  mxs.total = real_t(0);
  mxs.n_elements = 0;
  if (p == EmExtraProcess::kNone || !(ekin > real_t(0))) { return real_t(0); }
  int kind = -1;
  if (p == EmExtraProcess::kPhotonNuclear) { kind = kGammaNuclearThreshold; }
  if (p == EmExtraProcess::kElectronNuclear || p == EmExtraProcess::kPositronNuclear) {
    kind = kElectroNuclearThreshold;
  }
  if (kind >= 0 && !(ekin > emextra_threshold<real_t>(t, kind, mat_index))) {
    // At or below the threshold: every element's cross section is exactly zero. The partial
    // sums are left empty, which is what a caller that tried to draw a target from them would
    // be told - there is no target to draw.
    return real_t(0);
  }
  const EmExtraXsFn<real_t> fn = emextra_xs_fn<real_t>(t, p, ekin);
  const hxs::XsValue<real_t> v = hxs::store_compute_cross_section_fn<real_t>(
      fn, mat, hxs::nist_isotopes_of<real_t>(mat), mxs);
  return (v.value > real_t(0)) ? v.value : real_t(0);
}

/// The distance to the next lepto-nuclear interaction, mm, and the cross section it was drawn
/// with, for a stepper whose process is an ordinary discrete competitor: `electronNuclear`,
/// `positronNuclear`, `muonNuclear`.
///
/// ONE UNIFORM IF AND ONLY IF THE CROSS SECTION IS POSITIVE, and that conditional is the whole
/// of what keeps every electron below its material's threshold on the random stream it had
/// before P19. In B1 that is every electron of the 6 MeV gamma gate in water and air
/// (thresholds 7.296 and 6.599 MeV - oxygen and argon, each through CHIPS's `N = int(A) - Z`,
/// which makes oxygen O15) and all but the top 0.2 MeV of the spectrum in bone and A-150,
/// whose phosphorus and fluorine open at 5.593 and 5.606 MeV.
///
/// `__noinline__` for the reason `had::inelastic_length` gives: the partial sums stay on this
/// function's frame and are discarded, because `run_emextra` rebuilds them at the pre-step
/// energy the queue entry carries.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ real_t emextra_length(const EmExtraTables<real_t>& t,
                                                      EmExtraProcess p, int mat_index,
                                                      const data::Material<real_t>& mat,
                                                      real_t ekin, Rng& rng, real_t& xs_out) {
  hxs::MaterialXs<real_t> mxs{};
  const real_t xs = emextra_xs_per_volume<real_t>(t, p, mat_index, mat, ekin, mxs);
  xs_out = xs;
  if (!(xs > real_t(0))) { return real_t(1e30); }
  return -log(rng.uniform()) / xs;
}

/// The photo-nuclear term of the photon's total, 1/mm: `sigN` where the general process sums it,
/// zero where it does not.
///
/// Zero - and nothing evaluated - in zones 0 and 1 of the general process, at or below the
/// material's threshold, and with the process switched off. Otherwise the element sum
/// `G4GammaGeneralProcess::BuildPhysicsTable` computes through
/// `theGammaNuclear->GetCrossSectionDataStore()->ComputeCrossSection(dynParticle, material)`,
/// evaluated at the photon's energy rather than interpolated from the general process's zone-3
/// tables - the same choice P1 made for the other four terms, and the same interpolation-level
/// difference from Geant4.
///
/// NOT WHAT `step_gamma` READS IN ZONE 2 SINCE P21: there the general process applies table 9,
/// not this cross section (`gamma_general_p9`, docs/RISK.md V209 and P21's entry). It is read in
/// zone 3, where its share of the total goes to conversion (V207), and at every energy with the
/// general process off, where photonNuclear is an ordinary competitor.
template <typename real_t>
__host__ __device__ __noinline__ real_t photon_nuclear_xs(const EmExtraTables<real_t>& t,
                                                         GammaGeneralProcess mode,
                                                         int mat_index,
                                                         const data::Material<real_t>& mat,
                                                         real_t ekin) {
  if (gamma_nuclear_slot<real_t>(mode, ekin) == GammaNuclearSlot::kAbsent) { return real_t(0); }
  hxs::MaterialXs<real_t> mxs{};
  return emextra_xs_per_volume<real_t>(t, EmExtraProcess::kPhotonNuclear, mat_index, mat, ekin,
                                       mxs);
}

// =============================================================================================
// P21: table 9 - zone 2's photo-nuclear share, as G4GammaGeneralProcess builds and reads it
// =============================================================================================

/// `GetProbability(9)` in zone 2: `theHandler->GetVector(9, basedCoupleIndex)->LogVectorValue(
/// preStepKinEnergy, preStepLogE)` - the bin from the logarithm, the value LINEAR in E between its
/// two nodes, clamped to the end nodes outside (`xs/physics_vector.cuh`'s transcription of
/// `G4PhysicsVector::LogVectorValue`, whose `G4Log` is `std::log` on this platform). `preStepLogE`
/// is `G4DynamicParticle::GetLogKineticEnergy()`, which is `G4Log` of the same energy.
///
/// `__noinline__` FOR `photon_nuclear_xs`'s REASON, the other way round: it is reached only above
/// the material's table-9 threshold - which `step_gamma` tests inline first - and there a call
/// out of the 255-register kernel costs less than this function's registers inlined into every
/// photon step.
template <typename real_t>
__host__ __device__ __noinline__ real_t gamma_general_p9(const GammaGeneralTable9<real_t>& t,
                                                        int mat, real_t e) {
  hxs::PhysVec<real_t> pv;
  pv.e = t.e;
  pv.v = t.v + static_cast<long long>(mat) * kGammaGeneralZone2Nodes;
  pv.n = kGammaGeneralZone2Nodes;
  pv.edge_min = t.edge_min;
  pv.edge_max = t.edge_max;
  pv.inv_dbin = t.inv_dbin;
  pv.log_emin = t.log_emin;
  pv.type = hxs::kLogVector;
  return hxs::phys_vec_log_value<real_t>(pv, e, log(e));
}

/// True when `G4GammaGeneralProcess` could have selected photonNuclear for a photon of energy `e`
/// in material `mat`: the general process on, zone 2, and table 9 below 1 there. The drain asks it
/// of a photo-nuclear entry whose data store is EMPTY - the table's edge (this file's header) -
/// and books its tripwire only when the answer is no.
template <typename real_t>
__host__ __device__ inline bool gamma_general_selects_nuclear(const EmExtraTables<real_t>& t,
                                                             GammaGeneralProcess mode, int mat,
                                                             real_t e) {
  if (mode != GammaGeneralProcess::kOn || gamma_general_zone<real_t>(e) != 2 || t.p9 == nullptr) {
    return false;
  }
  if (mat < 0 || mat >= t.p9->n_materials || !(e > t.p9->threshold[mat])) { return false; }
  return gamma_general_p9<real_t>(*t.p9, mat, e) < real_t(1);
}

/// `cVector`'s 51 node energies, in `G4PhysicsLogVector`'s own order of operations
/// (G4PhysicsLogVector.cc, 11.1.1): the two edges stored as given, `Initialise()` setting
/// `invdBin = (idxmax + 1)/G4Log(edgeMax/edgeMin)` and `logemin = G4Log(edgeMin)`, and then
/// `binVector[i] = edgeMin*G4Exp(i/invdBin)` for the 49 inside. `G4Log` and `G4Exp` are
/// `std::log` and `std::exp` on this platform (G4Log.hh, G4Exp.hh `#ifdef WIN32`), and host code
/// computes this, so the nodes are Geant4's to the bit - which `tests/test_emextra_wiring.cu`
/// asserts against the running process's stored table.
template <typename real_t>
__host__ inline void gamma_general_zone2_grid(real_t* e, real_t& inv_dbin, real_t& log_emin) {
  const real_t emin = gamma_general_min_ee<real_t>();
  const real_t emax = gamma_general_min_mm<real_t>();
  e[0] = emin;
  e[kGammaGeneralZone2Bins] = emax;
  inv_dbin = real_t(kGammaGeneralZone2Bins) / std::log(emax / emin);
  log_emin = std::log(emin);
  for (int i = 1; i < kGammaGeneralZone2Bins; ++i) {
    e[i] = emin * std::exp(real_t(i) / inv_dbin);
  }
}

/// One node of table 9: the body of `BuildPhysicsTable`'s zone-2 loop, in its order of
/// operations -
///
///     G4double sum = sigComp + sigConv + sigPE + sigN;
///     val = (sigN > 0.0) ? (sigConv + sigComp + sigPE)/sum : 1.0;
///
/// with `sigComp`, `sigConv` and `sigPE` this port's (`em::gamma_macroscopic_xs`, no Rayleigh:
/// zone 2 sums none) and `sigN` the data store's (`gn->ComputeCrossSection`, which is
/// `emextra_xs_per_volume`). Geant4 takes the first two through `GetLambda`, i.e. off their
/// lambda tables, and this port evaluates the models - the one difference, measured in
/// section 8 of `tests/test_emextra_wiring.cu`.
///
/// @param t a view whose pointers are HOST pointers (`EmExtraHostTables::view()`).
template <typename real_t>
__host__ inline real_t gamma_general_table9_node(const EmExtraTables<real_t>& t, int mat_index,
                                                 const data::Material<real_t>& mat,
                                                 const data::PhotoElectricTable<real_t>& pe,
                                                 real_t e) {
  const em::GammaXS<real_t> xs = em::gamma_macroscopic_xs<real_t>(mat, e, &pe, nullptr);
  hxs::MaterialXs<real_t> mxs{};
  const real_t sig_n =
      emextra_xs_per_volume<real_t>(t, EmExtraProcess::kPhotonNuclear, mat_index, mat, e, mxs);
  const real_t sig_comp = xs.compton;
  const real_t sig_conv = xs.pair;
  const real_t sig_pe = xs.photoelectric;
  const real_t sum = sig_comp + sig_conv + sig_pe + sig_n;
  return (sig_n > real_t(0)) ? (sig_conv + sig_comp + sig_pe) / sum : real_t(1);
}

// =============================================================================================
// What the run counts
// =============================================================================================

/// The counters `HadronicWiring::emx_stats` holds - device-side, because the drain that runs
/// these interactions never reports back to the host until the run is over.
enum EmxStat : int {
  /// `photonNuclear` interactions `run_emextra_drain` ran (or refused by name).
  kEmxPhotoDrained = 0,
  /// `electronNuclear` + `positronNuclear` + `muonNuclear` interactions it ran.
  kEmxLeptoDrained = 1,
  /// Photon interactions at or above 100 MeV whose selection uniform fell in the photo-nuclear
  /// slice and became a CONVERSION, as `G4GammaGeneralProcess` 11.1.1 makes it (docs/RISK.md
  /// V207). The finding, counted: in QBBC this is how often a photon above 100 MeV would have
  /// reacted with a nucleus and did not.
  kEmxZone3Conversion = 2,
  /// P21: photons the general process put on a nucleus at an energy where the data store's
  /// cross section is EXACTLY zero - table 9's edge (this file's header), at or below the
  /// material's threshold and above its table's last 1.0 node. Each is applied as Geant4 applies
  /// it, on the material's first element; counted because it is a reaction no cross section
  /// asked for, and a run that has one should say so.
  kEmxTable9Edge = 3,
  kNumEmxStats = 4,
};

// =============================================================================================
// What the interaction kernel needs from the stepper's emitter
// =============================================================================================

/// `em.last_secondary` when the emitter has one, -1 when it does not.
///
/// A TRAIT AND NOT A MEMBER ACCESS, because `step_gamma` and `step_lepton` are also instantiated
/// with `src/host/b1_gpu_sched.cu`'s `SchedEmitter`, which carries no secondary arena and never
/// enqueues (it passes no wiring). A plain `em.last_secondary` in the enqueue branch would stop
/// that driver compiling for a branch it cannot reach; the lead's note on P14c's `books` member
/// is the record of that driver's emitter being edited by hand for exactly this reason.
template <typename E, typename = void>
struct EmitterHasLastSecondary : std::false_type {};
template <typename E>
struct EmitterHasLastSecondary<E, std::void_t<decltype(std::declval<E&>().last_secondary)>>
    : std::true_type {};

template <typename E>
__host__ __device__ inline int emitter_last_secondary(const E& em) {
  if constexpr (EmitterHasLastSecondary<E>::value) {
    return em.last_secondary;
  } else {
    return -1;
  }
}

}  // namespace g4gpu::had
