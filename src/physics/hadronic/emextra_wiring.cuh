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
// THE REST OF THE PHOTON'S STEP IS NOT TOUCHED, AND THAT IS A DECISION WITH A NUMBER ON IT.
// The port's photon is P1's: four cross sections evaluated from the models every step, one
// length uniform and one selection uniform walked as Compton, conversion, photoelectric,
// Rayleigh. The general process differs from that in two more ways than photonNuclear -
// Rayleigh is ABSENT above 2 m_e (zones 2 and 3 sum no `sigR`), and its shares are its own
// tables, interpolated linearly between 51 nodes from 2 m_e to 100 MeV - and neither is P19's
// to change. The first moves a photon in water by 4.3e-4 of its interactions at 1.5 MeV and
// 5.7e-5 at 6 MeV (the running Geant4 counts no Rayleigh scatter above 2 m_e in a million where
// its share is 432 and 57, docs/RISK.md V208), and the brief's bit-identity is exactly the
// property that forbids moving it in a package about nuclei. The second is where the
// photo-nuclear share itself parts from its cross section: across the giant resonance the
// interpolated share is 16% under sigN/sum at 22 MeV in bone and 8% over at 20 MeV in water,
// the running Geant4 follows the table to within statistics, and this port follows the cross
// section, which is what the P19 brief specifies (docs/RISK.md V209 has both and the recipe for
// the table).
// The photo-nuclear slice is placed where the general process places it RELATIVE TO THE ONE
// UNIFORM: it is the TOP slice of `q` in zone 2 in both codes (`q > P9` there, `q*total >=
// total_em` here), so for a given `q` the two agree on whether the photon reacts with a
// nucleus - which is the sense in which the selection is on Geant4's stream.
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
/// `150*CLHEP::keV`, `2*CLHEP::electron_mass_c2`, `100*CLHEP::MeV`. The zone edges.
template <typename real_t>
__host__ __device__ inline constexpr real_t gamma_general_min_pe() { return real_t(0.150); }
template <typename real_t>
__host__ __device__ inline constexpr real_t gamma_general_min_ee() {
  return real_t(2) * units::electron_mass_c2<real_t>();
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
  return em::GammaProcess::kRayleigh;
}

// =============================================================================================
// The device tables
// =============================================================================================

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
};

/// The threshold for one (kind, material), or zero when none was uploaded - zero meaning
/// "evaluate at every energy", which is the answer that cannot be wrong.
template <typename real_t>
__host__ __device__ inline real_t emextra_threshold(const EmExtraTables<real_t>& t, int kind,
                                                    int mat) {
  if (t.threshold == nullptr || mat < 0 || mat >= t.n_materials) { return real_t(0); }
  return t.threshold[kind * t.n_materials + mat];
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
    switch (process) {
      case EmExtraProcess::kPhotonNuclear:
        if (tables->gamma == nullptr) { return {real_t(0), hxs::XsRefusal::kNone}; }
        return hxs::pxs_element_xs<real_t>(*tables->gamma, ekin, loge, Z);
      case EmExtraProcess::kElectronNuclear:
      case EmExtraProcess::kPositronNuclear:
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
/// evaluated at the photon's energy rather than interpolated from the general process's zone-2
/// and zone-3 tables - the same choice P1 made for the other four terms, and the same
/// interpolation-level difference from Geant4.
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
// What the run counts
// =============================================================================================

/// The three counters `HadronicWiring::emx_stats` holds - device-side, because the drain that
/// runs these interactions never reports back to the host until the run is over.
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
  kNumEmxStats = 3,
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
