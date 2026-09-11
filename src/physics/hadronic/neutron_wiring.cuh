// P8d's neutron wiring: which cross section and which final state each sub-process of
// G4NeutronGeneralProcess uses, and one function per sub-process that runs it.
//
// The neutron's counterpart to `elastic_wiring.cuh`, and it owns no physics either. P2 has the
// cross sections (`xs/`), P5 has the framework and the elastic final states (`process.cuh`,
// `elastic/`), P7 has the capture (`capture/`), P3 has the cascade under it; this file is where
// `step_neutral` asks them the right question in the right order.
//
// ---------------------------------------------------------------------------------------------
// THE TABLE, from G4HadProcesses::BuildNeutronElastic and BuildNeutronInelasticAndCapture
// (11.1.1, source/physics_lists/util/src/G4HadProcesses.cc), both on the `useNeutronGeneral`
// branch:
//
//   sub-process   cross section          final-state model            where
//   elastic       G4NeutronElasticXS     G4ChipsElasticModel          xs/particlexs.cuh,
//                                                                    elastic/chips_elastic.cuh
//   inelastic     G4NeutronInelasticXS   BIC / BERT / FTFP            REFUSED BY NAME (P9-P11)
//   capture       G4NeutronCaptureXS     G4NeutronRadCapture          capture/
//
// THE NEUTRON'S ELASTIC IS NOT THE PROTON'S, AND THAT IS THE FIRST THING TO GET RIGHT.
// `G4HadronElasticPhysics::ConstructProcess` gives the proton `G4BGGNucleonElasticXS` and the
// neutron `G4NeutronElasticXS` - a G4PARTICLEXS data set, not a Glauber-Gribov/Barashenkov
// composite - while both get `G4ChipsElasticModel`. So the neutron shares the MODEL with the
// proton and not the data set, which is why this file exists beside `elastic_wiring.cuh` rather
// than adding a row to it: `had::ElasticXsFn` binds an energy and no logarithm, and every
// G4PARTICLEXS lookup needs `loge` as well (`G4DynamicParticle::GetLogKineticEnergy`). Threading
// a second energy through P8b's three entry points would put one `log()` on every step of every
// charged hadron for a branch only the neutron takes.
//
// WHY THERE IS NO INTEGRAL REJECTION HERE, AND IT IS A FACT ABOUT THE CHARGE
// `G4HadronicProcess::BuildPhysicsTable` guards the whole integral-approach setup with
// `if (charge != 0.0 && useIntegralXS && !isLepton && ok)`, so a neutron's elastic and capture
// sub-processes are both `fHadNoIntegral`: `PostStepDoIt` draws no rejection uniform and, more
// importantly, does NOT recompute the cross section - so the element is drawn from the partial
// sums the CALLER left behind. For a neutral particle the pre- and post-step energies are the
// same number (there is no continuous loss), so "the caller's" and "recomputed" would agree
// numerically; the draw COUNT would not, and that is the random stream.
//
// WHO COMPUTES THE PARTIAL SUMS, AND ONLY WHEN
// `G4NeutronGeneralProcess::PostStepDoIt` ends with
//
//     if(fCurrMat->GetNumberOfElements() > 1) {
//       fCurrentXSS->ComputeCrossSection(track.GetDynamicParticle(), fCurrMat);
//     }
//     return fSelectedProc->PostStepDoIt(track, step);
//
// - the chosen sub-process's own data store, and ONLY for a compound. With one element
// `SampleZandA` takes element 0 without looking at a partial sum and draws no uniform, so the
// call would be work with no answer attached. Both callers here pass a `MaterialXs` and say
// which case they are in.
//
// ---------------------------------------------------------------------------------------------
// `__noinline__`, ON BOTH, AND IT IS WHAT MAKES THE ENGINE COMPILE
//
// docs/RISK.md V55 is the entry: wiring `hadElastic` into `step_hadron` killed ptxas with an
// access violation, and the fix was to stop inlining rather than to simplify the physics. Its
// last paragraph names this package - "the neutron general process is the next thing to be wired
// into a stepper and it drags the whole de-excitation chain into this file through
// `capture/capture_process.cuh`; whoever does it should put `__noinline__` on the capture cascade
// before measuring anything".
//
// Measured with V55's one-kernel reproducer on `run_step_neutral<double, kNeutron>`, which is
// the only affordable way to iterate on `transport_run.cu` - the reproducer compiles in 75 s
// against the translation unit's 22 minutes (P8c recorded ninety, with three Geant4 cmake builds
// running beside it):
//
//   baseline, main at 2a6b379, no neutron physics  3408 B stack,  88/52 B spill, 255 registers
//   the elastic sub-process alone                  3056 B
//   both sub-processes, capacity 32               13120 B
//   both sub-processes, capacity 16                7472 B       296/516, 255 registers
//
// The 5,648 bytes between the last two rows are `kNeutronCaptureSecondaryCap` and nothing else;
// the ELASTIC branch costs the frame almost nothing, because its `HadFinalState<real_t, 1>` is
// 56 bytes (see `neutron_elastic_apply`). So the `__noinline__` on the elastic side is not what
// saves the frame - it is what stops G4ChipsElasticModel's tables being a second copy of what
// `step_hadron` already inlines - and the one on the capture side is both.
//
// And it is the right answer for the hot path for the same reason it was there: the capture
// cascade walks a nuclide's level scheme and the elastic branch carries G4ChipsElasticModel's
// 52-parameter tables, and neither runs on a step that does not interact. The mean free paths
// make that a quantity rather than a hope - measured in `tests/test_neutron_general.cu`, a
// 10 MeV neutron's total mean free path in AIR is about 62 m (at a half-width of 1 km, 3,249 of
// 200,000 tracks left the box), and the capture sub-process is 3.4e-5 of the interactions there
// - 5.1e-5 in water, 6.7e-5 in bone, 3.3e-4 in lead. So the cascade runs on about one
// interacting step in twenty thousand.
#pragma once

#include <cmath>

#include "core/particle.cuh"
#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/level_data.cuh"
#include "data/materials.cuh"
#include "physics/hadronic/capture/capture_process.cuh"
#include "physics/hadronic/elastic/chips_elastic.cuh"
#include "physics/hadronic/elastic/elastic_process.cuh"
#include "physics/hadronic/elastic/hadron_elastic.cuh"
#include "physics/hadronic/neutron_general_xs.cuh"
#include "physics/hadronic/process.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/projectile.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

namespace g4gpu::had {

namespace nel = g4gpu::physics::hadronic::elastic;
namespace ncap = g4gpu::physics::hadronic::capture;
namespace nhp = g4gpu::physics::hadronic;

/// The two per-process cross sections the neutron needs on the device, as
/// `G4NeutronElasticXS` and `G4NeutronCaptureXS` - a null pointer being "no such process".
///
/// BOTH STAGES READ THEM AND FOR DIFFERENT REASONS, which is why they are here rather than
/// inside the stage switch. In `HadronicStage::kStage1` each is a whole process with its own
/// interaction length, because that is what Geant4 runs with `EnableNeutronGeneralProcess`
/// false. In `kFinal` the interaction length comes from the general table instead and these are
/// what `SampleZandA` draws a target from - the sub-process's own data store, which is the one
/// `G4NeutronGeneralProcess::PostStepDoIt` calls `ComputeCrossSection` on before delegating.
///
/// `host/neutron_upload.cuh` fills them.
template <typename real_t>
struct NeutronSubTables {
  const hadronic::xs::PxsDataSet<real_t>* elastic = nullptr;  ///< G4NeutronElasticXS
  const hadronic::xs::PxsDataSet<real_t>* capture = nullptr;  ///< G4NeutronCaptureXS
};

/// `G4HadProjectile` for a neutron at this kinetic energy.
///
/// The mass is `units::neutron_mass_c2`, i.e. the DEFINITION's PDG mass, because that is what
/// `G4HadProjectile::Initialise` takes from the track - and `xs::neutron()`'s own mass is the
/// `0.939565*GeV` literal the cross sections are written against, which is a different number in
/// the seventh digit. The two are used for the two different things they are, as
/// `tests/test_capture.cu` and P2's `xs/projectile.cuh` each say of their own.
template <typename real_t>
__host__ __device__ inline nhp::HadProjectile<real_t> neutron_projectile(real_t ekin) {
  nhp::HadProjectile<real_t> p;
  p.pdg = 2112;
  p.baryon_number = 1;
  p.charge = real_t(0);
  p.mass = units::neutron_mass_c2<real_t>();
  p.kin_energy = ekin;
  return p;
}

/// One sub-process's macroscopic cross section, 1/mm, and the per-element partial sums a target
/// draw would need. `G4CrossSectionDataStore::ComputeCrossSection` with one registered data set.
///
/// Zero, with `mxs` zeroed, for a null data set - which is the state a run whose
/// `G4PARTICLEXSDATA` could not be resolved is in, and behaves exactly as a species with no such
/// process does: an infinite interaction length and no uniform drawn.
///
/// `__noinline__` for the reason the file header gives: inlined, this is
/// `G4NeutronElasticXS::ComputeCrossSectionPerElement` and `G4NeutronCaptureXS`'s 1/sqrt(E)
/// low-energy arm and the Glauber-Gribov hand-over above the table, in the body of
/// `run_step_neutral`, twice.
template <typename real_t>
__host__ __device__ __noinline__ real_t neutron_sub_xs_per_volume(
    const hadronic::xs::PxsDataSet<real_t>* ds, const data::Material<real_t>& mat, real_t ekin,
    real_t loge, hadronic::xs::MaterialXs<real_t>& mxs) {
  if (ds == nullptr) {
    mxs.total = real_t(0);
    mxs.n_elements = 0;
    return real_t(0);
  }
  const hadronic::xs::XsValue<real_t> v = hadronic::xs::store_compute_cross_section<real_t>(
      *ds, ekin, loge, mat, hadronic::xs::nist_isotopes_of<real_t>(mat), mxs);
  return (v.value > real_t(0)) ? v.value : real_t(0);
}

/// What one elastic sub-process PostStepDoIt did to a neutron.
///
/// The same shape as `had::ElasticStepOutcome` minus the two fields that are about a charged
/// hadron - there is no integral rejection for a neutral particle and no
/// `G4ElasticHadrNucleusHE` fall-back, because the neutron's model is CHIPS - plus
/// `stopped_but_alive`, which is the one place this port knowingly differs from Geant4 and so is
/// reported rather than folded into `primary_survives`.
template <typename real_t>
struct NeutronElasticOutcome {
  bool interacted = false;
  Vec3<real_t> dir{real_t(0), real_t(0), real_t(1)};
  real_t energy = 0;             ///< the primary's new kinetic energy
  bool primary_survives = true;
  /// `efinal == 0` and the neutron HAS an at-rest process (`G4Decay`), so
  /// `G4HadronElasticProcess::PostStepDoIt` proposes fStopButAlive and Geant4 keeps the track.
  /// This transport kills it - see the note at `neutron_elastic_apply`.
  bool stopped_but_alive = false;
  real_t edep = 0;               ///< local AND non-ionizing - see elastic_process.cuh note 5
  bool emit_recoil = false;
  int recoil_z = 0, recoil_a = 0;
  real_t recoil_ekin = 0;
  Vec3<real_t> recoil_dir{real_t(0), real_t(0), real_t(1)};
  int dropped_secondaries = 0;
  /// (Z, A) SampleZandA drew, so a test can assert the target and not only the kinematics.
  int target_z = 0, target_a = 0;
  /// G4ChipsElasticModel fell back to Gheisha inside SampleInvariantT.
  bool chips_fell_back = false;
  /// The Chips tables have no row for this (projectile, Z, A). Reported rather than silently
  /// taken as a Gheisha scatter, because the fall-back and the absence are different facts.
  bool chips_unsupported = false;
};

/// The whole elastic sub-process: `SampleZandA` on G4NeutronElasticXS, then
/// `G4HadronElasticProcess::PostStepDoIt` with `G4ChipsElasticModel`.
///
/// ZERO UNIFORMS ARE DRAWN BEFORE THE TARGET, which is the difference from
/// `had::elastic_apply`: `fXSType` is `fHadNoIntegral` for a neutron, so the rejection that
/// costs a charged hadron one uniform and one whole recomputation of the cross section does not
/// happen. `elastic_post_step_do_it` is handed `kNoIntegral` and the two cross sections it would
/// have compared are passed equal, so the comparison cannot fire whatever the type.
///
/// THE ONE KNOWN DIVERGENCE, stated where it happens. When the final energy is zero - which an
/// elastic scatter off hydrogen at 180 degrees produces - Geant4 asks whether the particle has
/// at-rest processes, and the neutron does (`G4Decay` is a G4VRestDiscreteProcess and sits on the
/// neutron's at-rest vector), so it proposes `fStopButAlive` and the track survives at zero
/// energy. This transport cannot step that track: `TrackState::advance` moves the clock by
/// `L/(beta c)` and beta is zero, so a zero-energy neutron would either not age or age by an
/// undefined amount, and the only at-rest process it has is a decay with an 880 s lifetime
/// against this process's own 10 us cut. So the port kills it, with no deposit - which is where
/// Geant4's track ends up as well, one step later and via the time cut. `stopped_but_alive`
/// counts it so the size of the divergence is a number and not an argument.
///
/// @param mxs must have been filled by `neutron_sub_xs_per_volume` on the ELASTIC data set at
///        THIS energy and in THIS material, or element 0 is chosen by another energy's partial
///        sums and nothing complains. For a single-element material it is not read at all, which
///        is `SampleZandA`'s own `if(1 < nElements)`.
template <typename real_t, typename Rng>
__host__ __device__ __noinline__ NeutronElasticOutcome<real_t> neutron_elastic_apply(
    const hadronic::xs::PxsDataSet<real_t>& el, const data::Material<real_t>& mat, real_t ekin,
    real_t loge, const Vec3<real_t>& in_dir, real_t range_cut_mm,
    const hadronic::xs::MaterialXs<real_t>& mxs, Rng& rng) {
  NeutronElasticOutcome<real_t> out;
  out.dir = in_dir;
  out.energy = ekin;

  const hadronic::xs::TargetZA tgt = hadronic::xs::store_sample_za_rng<real_t>(
      hadronic::xs::pxs_xs_functions<real_t>(el, ekin, loge), mat,
      hadronic::xs::nist_isotopes_of<real_t>(mat), mxs, rng);
  out.target_z = tgt.z;
  out.target_a = tgt.a;
  if (tgt.a <= 0 || tgt.z <= 0) { return out; }

  nhp::HadNucleus target;
  target.z = tgt.z;
  target.a = tgt.a;

  const real_t recoil_threshold = nel::proton_recoil_cut_energy<real_t>(range_cut_mm);
  // PLAIN LAMBDAS, not `__host__ __device__` ones, for the reason `elastic_wiring.cuh` gives:
  // nvcc gives a lambda defined inside a `__host__ __device__` function the enclosing
  // function's execution space, and an explicit annotation would need `--extended-lambda`.
  auto nuclear_mass = [](int z, int a) { return data::nuclear_mass<real_t>(a, z); };
  auto recoil_species = [](int z, int a, int* pdg, real_t* mass) {
    *pdg = nhp::pdg_nuclear_code(z, a);
    *mass = data::nuclear_mass<real_t>(a, z);
  };

  // ONE, AND IT IS PROVABLE: `hadron_elastic_apply_yourself` calls `add_secondary` exactly once
  // and `G4HadronElasticProcess::PostStepDoIt` looks only at `GetSecondary(0)` (note 4 of
  // elastic/elastic_process.cuh). 56 bytes of frame where the package default of 8 would be 450.
  nhp::HadFinalState<real_t, 1> fs;
  bool chips_fb = false, chips_unsupported = false;
  auto apply = [&](const nhp::HadProjectile<real_t>& p, const nhp::HadNucleus& n, real_t thr,
                   Rng& r, nhp::HadFinalState<real_t, 1>* o) {
    auto sampler = [&](int pdg, real_t plab, int z, int a, real_t p_local_tmax,
                       Rng& rr) -> real_t {
      // G4ChipsElasticModel::SampleInvariantT, whose Gheisha fall-back is INSIDE it - so the
      // uniforms of both attempts are consumed there, which is where Geant4 consumes them.
      return nel::chips_sample_invariant_t<real_t>(pdg, plab, z, a, p_local_tmax, nuclear_mass,
                                                    rr, &chips_fb, &chips_unsupported);
    };
    (void)nel::hadron_elastic_apply_yourself<real_t, 1>(p, n, sampler, nuclear_mass,
                                                         recoil_species, thr, 0, r, o);
  };

  // `has_at_rest_processes = true`: the neutron's at-rest vector holds `G4Decay`. The status
  // that produces - fStopButAlive - is reported in `stopped_but_alive` and not honoured; see
  // the note above.
  const nel::ElasticStepResult<real_t> r = nel::elastic_post_step_do_it<real_t, 1>(
      neutron_projectile<real_t>(ekin), target, in_dir, real_t(1), true,
      /*has_at_rest_processes=*/true, nhp::HadXsType::kNoIntegral, real_t(1), real_t(1),
      recoil_threshold, apply, rng, &fs);

  out.interacted = r.interacted;
  out.dir = r.momentum_direction;
  out.energy = r.energy;
  out.primary_survives = (r.status == nhp::TrackStatusChange::kAlive);
  out.stopped_but_alive = (r.status == nhp::TrackStatusChange::kStopButAlive);
  out.edep = r.local_energy_deposit;
  out.dropped_secondaries = r.dropped_secondaries + fs.secondary_overflow;
  out.chips_fell_back = chips_fb;
  out.chips_unsupported = chips_unsupported;
  if (r.n_secondaries > 0) {
    out.emit_recoil = true;
    out.recoil_z = r.recoil.z;
    out.recoil_a = r.recoil.a;
    out.recoil_ekin = r.recoil.kin_energy;
    out.recoil_dir = r.recoil.direction;
  }
  return out;
}

/// `G4NeutronRadCapture`'s final state, as many secondaries as the cascade made, and the
/// arithmetic the generic process wraps it in.
///
/// **16, AND IT COSTS 314 BYTES OF KERNEL FRAME PER UNIT, MEASURED.**
///
/// The capacity is paid TWICE - once for the model's `HadFinalState` scratch and once for
/// `fill_result`'s `HadronicStepResult` - and `sizeof(HadSecondary<double>)` is 80 bytes, so the
/// two arrays plus `capture_final_state`'s `pdg_mass[kCap]` are 168 bytes a slot on paper.
/// Measured with V55's one-kernel reproducer on `run_step_neutral<double, kNeutron>` the real
/// figure is about twice that, because ptxas keeps parts of both arrays live at once - 353 bytes
/// of kernel frame per unit of capacity, over the range that was measured:
///
///     main at 2a6b379, no neutron physics at all        3408 B stack frame
///     the elastic sub-process only                      3056 B
///     both, capacity 32                                13120 B
///     both, capacity 16                                 7472 B
///
/// against the 16384-byte limit `TransportEngine::Upload` sets. `tests/test_capture_device.cu`
/// ran 448 captures at 32 and P7's statistical oracle 20,000 per (target, energy); neither
/// reports a maximum, so `tests/test_neutron_general.cu` measures one - the longest cascade over
/// every element of B1's four materials - and asserts it is under this, with
/// `HadronicRefusal::kCaptureOverflow` as the tripwire in transport. A capacity chosen for the
/// frame rather than for the physics is only defensible if the physics is then measured against
/// it, which is what that section is for.
constexpr int kNeutronCaptureSecondaryCap = 16;

/// What one capture sub-process did.
template <typename real_t>
struct NeutronCaptureOutcome {
  real_t edep = 0;               ///< G4HadFinalState::GetLocalEnergyDeposit
  int n_emitted = 0;             ///< secondaries that became tracks or bookings
  int overflow = 0;              ///< the cascade made more than the final state can hold
  int unmapped = 0;              ///< a secondary PDG code core/particle.cuh has no row for
  int target_z = 0, target_a = 0;
  int n_gammas = 0;
  int residual_z = 0, residual_a = 0;
  bool refused_resample_limit = false;
  /// `CaptureRefusal` as an int, so a caller can book it without this file depending on the
  /// refusal ledger (which lives in `wiring.cuh`, which includes this one).
  int capture_refusal = 0;
};

/// The whole capture sub-process: `SampleZandA` on G4NeutronCaptureXS, then
/// `G4NeutronCaptureProcess`'s PostStepDoIt - P7's `capture_final_state` - and then the
/// secondaries into the track pool.
///
/// THE EMISSION IS INSIDE THIS FUNCTION AND THAT IS THE POINT OF THE FUNCTION. Both of the
/// 32-entry secondary arrays live in this frame; returning them would put 5 KB of them in
/// `run_step_neutral`'s, which is the frame docs/RISK.md V22 and V55 are about.
///
/// @param mxs as for the elastic sub-process, on the CAPTURE data set.
/// @param em the emitter, positioned by the caller. A capture emits gammas, sometimes an
///        internal-conversion electron, and the residual NUCLEUS - which is why
///        `push_nucleus` appears here and not `push`: a nuclide is the one secondary whose
///        species does not say what it is (core/track_buffer.cuh).
template <typename real_t, typename Rng, typename Emitter>
__host__ __device__ __noinline__ NeutronCaptureOutcome<real_t> neutron_capture_apply(
    const hadronic::xs::PxsDataSet<real_t>& cap, const data::Material<real_t>& mat, real_t ekin,
    real_t loge, const Vec3<real_t>& in_dir, real_t global_time, real_t weight,
    const data::LevelTable& levels, const hadronic::xs::MaterialXs<real_t>& mxs, Rng& rng,
    Emitter& em, int event) {
  NeutronCaptureOutcome<real_t> out;

  const hadronic::xs::TargetZA tgt = hadronic::xs::store_sample_za_rng<real_t>(
      hadronic::xs::pxs_xs_functions<real_t>(cap, ekin, loge), mat,
      hadronic::xs::nist_isotopes_of<real_t>(mat), mxs, rng);
  out.target_z = tgt.z;
  out.target_a = tgt.a;
  if (tgt.a <= 0 || tgt.z <= 0) { return out; }

  nhp::HadNucleus target;
  target.z = tgt.z;
  target.a = tgt.a;

  constexpr int kCap = kNeutronCaptureSecondaryCap;
  nhp::HadFinalState<real_t, kCap> scratch;
  // `has_at_rest_processes` is read by `fill_result` only when the model leaves a positive-energy
  // primary, and capture never does - `SetStatusChange(stopAndKill)` is the second line of
  // `ApplyYourself`, and that branch wins first (note 6 of capture/capture_process.cuh). The
  // neutron's own answer is `true` (`G4Decay`), passed rather than assumed.
  const ncap::CaptureStepResult<real_t, kCap> r = ncap::capture_final_state<real_t, kCap, kCap>(
      neutron_projectile<real_t>(ekin), target, in_dir, global_time, weight,
      /*has_at_rest_processes=*/true, levels, rng, &scratch);

  out.edep = r.change.local_energy_deposit;
  out.overflow = r.change.secondary_overflow;
  out.refused_resample_limit = r.refused_resample_limit;
  out.capture_refusal = static_cast<int>(r.info.refused);
  out.n_gammas = r.info.n_gammas;
  out.residual_z = r.info.residual_z;
  out.residual_a = r.info.residual_a;

  for (int i = 0; i < r.change.n_secondaries; ++i) {
    const nhp::HadSecondary<real_t>& s = r.change.secondaries[i];
    if (s.a > 0) {
      // The residual, and any fragment the cascade split off. `push_nucleus` maps (Z, A) onto
      // the five light species that carry their own definition or onto `kGenericIon` with the
      // nuclide on the track, which is what P8c's transport reads.
      em.push_nucleus(s.z, s.a, s.direction, s.kin_energy, event);
      ++out.n_emitted;
      continue;
    }
    const ParticleType t = particle_type_of_pdg(s.pdg);
    if (t == ParticleType::kNumTypes) {
      // A PDG code with no row in the enum. Not reachable from a radiative capture - every
      // secondary is a gamma, an internal-conversion electron or a nucleus - and counted
      // rather than dropped, so that a cascade that starts emitting something else is loud.
      ++out.unmapped;
      continue;
    }
    em.push(t, s.direction, s.kin_energy, event);
    ++out.n_emitted;
  }
  return out;
}

}  // namespace g4gpu::had
