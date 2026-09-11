// Single-step physics for the track-parallel scheduler.
//
// Restructured from "follow one track to
// completion" into "advance one track by one step". That is what removes the per-thread
// secondary stack and the loop-carried state that drove register pressure to 108 in the
// thread-per-event kernel.
#pragma once
#include "core/step_report.cuh"
#include "core/track_buffer.cuh"
#include "data/materials.cuh"
#include "geometry/navigator.cuh"
#include "geometry/safety.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/fluctuation.cuh"
#include "physics/em/ion_fluctuation.cuh"
#include "physics/em/hadron_delta.cuh"
#include "physics/em/hadron_range.cuh"
#include "physics/em/nuclear_stopping.cuh"
#include "physics/em/coulomb_scattering.cuh"
#include "physics/em/wentzel_msc.cuh"
#include "physics/em/gamma_processes.cuh"
#include "physics/em/klein_nishina.cuh"
#include "physics/em/annihilation.cuh"
#include "physics/em/pair_production.cuh"
#include "physics/em/urban_msc.cuh"
#include "physics/decay/decay.cuh"
#include "physics/hadronic/neutron_general_xs.cuh"
#include "physics/hadronic/wiring.cuh"
#include "physics/scene.cuh"
#include "render/trajectory.cuh"

namespace g4gpu {

/// `G4CoulombScattering::PostStepDoIt`, applied to a track. The wiring half of P14's model.
///
/// Called from `step_lepton` and `step_hadron` on the POST-step state, which is what
/// `G4VEmProcess::PostStepDoIt` reads: `G4Step::UpdateTrack` has already run, so
/// `track.GetKineticEnergy()` is the energy after the continuous loss. Written once for both
/// because the sequence is the same and the ORDER inside it is Geant4's.
///
/// THE INTEGRAL-CROSS-SECTION REJECTION IS HERE AND IT IS NOT OPTIONAL.
///
/// `G4CoulombScattering::InitialiseProcess` calls `SetCrossSectionType(fEmIncreasing)` when
/// MscThetaLimit is pi (G4CoulombScattering.cc:102), which it is in option0. So
/// `G4VEmProcess::PostStepDoIt` runs
///
///     const G4double lx = std::max(GetCurrentLambda(finalT, logFinalT), 0.0);
///     if(preStepLambda*G4UniformRand() >= lx) { return &fParticleChange; }
///
/// - the mean free path was drawn with the cross section at the START of the step and the
/// interaction is then thrown away with probability `1 - lx/preStepLambda`, which makes the
/// effective rate the cross section at the step's END. Without it a charged hadron that lost a
/// lot of energy over a long step would scatter at the wrong rate, and the rate would be wrong
/// in the direction of too many scatters. `hadronic/process.cuh::integral_xs_rejects` is the
/// same mechanism for the hadronic processes.
///
/// The one thing this does NOT reproduce is Geant4's CACHING of `preStepLambda`:
/// `ComputeIntegralLambda`'s fEmIncreasing branch keeps a lambda computed at up to 1/0.8 of the
/// current energy until the energy has fallen by 20%, so Geant4's `preStepLambda` can be larger
/// than the true pre-step value and its rejection correspondingly harder. Both give the same
/// net rate `lx`; what differs is the number of uniforms consumed, and this port re-draws every
/// interaction length every step anyway (see `had::decay_in_flight_length`'s header for why
/// that is the same distribution and not an approximation).
///
/// @param xs_at_step_start the cross section the step's interaction length was drawn with,
///        1/mm - `preStepLambda`.
template <typename real_t, typename Rng, typename Emitter>
__host__ __device__ __noinline__ void coulomb_apply(const Scene<real_t>& s, TrackState<real_t>& p,
                                     ParticleType type, int mat, real_t xs_at_step_start,
                                     Rng& rng, Emitter& em, real_t& edep,
                                     StepReport<real_t>& rep) {
  const data::Material<real_t>& mm = s.materials[mat];
  // One value, used as `cutEnergy` in SetupTarget and as the recoil threshold: in 11.1.1 they
  // are literally the same cuts vector, the PROTON's, because the process declares
  // SetSecondaryParticle(G4Proton::Proton()). See em/coulomb_scattering.cuh's file header.
  const real_t cut = em::coulomb_secondary_cut(s.range_cut);

  const real_t xs_now =
      em::coulomb_xs_per_volume(mm, type, p.ekin, cut, real_t(-1), real_t(-1));
  // Geant4's own test, including its direction at equality.
  if (xs_at_step_start * rng.uniform() >= xs_now) { return; }

  const auto r = em::coulomb_fire(mm, type, p.ekin, p.dir, cut, rng);
  if (!r.fired) { return; }

  p.dir = r.dir;
  p.ekin = r.final_t;
  // ProposeLocalEnergyDeposit AND ProposeNonIonizingEnergyDeposit, with the same value: a
  // sub-threshold nuclear recoil is entirely non-ionizing, so a scorer that separates the two
  // sees it there. G4Step::GetNonIonizingEnergyDeposit is part of edep, not additional to it.
  if (r.edep > real_t(0)) {
    if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += r.edep; }
    rep.non_ionizing += r.edep;
  }
  if (r.emit_ion) {
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    // Hydrogen recoils as a proton and is transported; everything heavier maps to
    // `kGenericIon`, which has no kernel, so `BufferEmitter::push` counts it by name with its
    // kinetic energy. That is a hole in the answer with a number attached rather than a silent
    // drop - see EmitterBooks::refused_energy, which P8 added for exactly this shape of
    // secondary.
    em.push_nucleus(r.ion_z, r.ion_a, r.ion_dir, r.ion_ekin, p.event);
  }
}

/// Advances one photon by a single step.
/// @param edep energy deposited in the scoring volume by this step, MeV
/// @param rep what the step did, beyond depositing energy: the true path length, the process
///        that ended it, the material, the safety. See core/step_report.cuh. Every field on it
///        is a value this function already computes; nothing here is calculated for its sake.
/// @return true if the photon is still alive and should be requeued
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_gamma(const Scene<real_t>& s, TrackState<real_t>& p, Rng& rng,
                                  Emitter& em, real_t& edep, StepReport<real_t>& rep,
                                  vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  rep = StepReport<real_t>{};
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld) { return false; }

  if (p.ekin < em::kPhotonAbsorbCut<real_t>()) {
    rep.material = geom::material_at(s.geometry, p.volume, p.pos);
    rep.status = StepStatus::fStopAndKill;
    rep.process = ProcessId::fBelowTrackingCut;
    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = p.ekin; }
    return false;
  }

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
  rep.material = mat;
  const auto xs =
      em::gamma_macroscopic_xs(s.materials[mat], p.ekin,
                               s.processes.photoelectric ? s.photoelectric : nullptr,
                               s.processes.rayleigh ? s.rayleigh : nullptr,
                               s.processes.compton, s.processes.pair_production);

  int next_volume = geom::kOutsideWorld;
  const real_t d_boundary =
      geom::step_to_boundary(s.geometry, p.volume, p.pos, p.dir, next_volume);

  const real_t s_int =
      (xs.total > real_t(0)) ? -log(rng.uniform()) / xs.total : geom::kInfinity<real_t>();

  // Streaming to the boundary counts as one step; the photon is requeued.
  if (s_int >= d_boundary) {
    rep.true_length = d_boundary + geom::kPushDistance<real_t>();
    rep.status = StepStatus::fGeomBoundary;
    rep.process = ProcessId::fTransportation;
    p.pos = p.pos + (d_boundary + geom::kPushDistance<real_t>()) * p.dir;
    traj.add(pos_before, p.pos, ParticleType::kGamma, p.event, p.rng_key);
    p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
    return p.volume != geom::kOutsideWorld;
  }

  // A discrete process fired, so the step ended on physics rather than on geometry. Which one
  // is recorded in each branch below rather than from `proc` here, because a branch can be
  // skipped when its data table is absent and the one that actually ran is what a stepping
  // action must be told about.
  rep.true_length = s_int;
  rep.status = StepStatus::fPostStepDoItProc;
  p.pos = p.pos + s_int * p.dir;
  traj.add(pos_before, p.pos, ParticleType::kGamma, p.event, p.rng_key);
  em.pos = p.pos;
  em.volume = p.volume;
  em.event = p.event;

  const auto proc = em::select_gamma_process(xs, rng.uniform());

  if (proc == em::GammaProcess::kCompton) {
    rep.process = ProcessId::fCompton;
    const auto r = em::sample_klein_nishina<real_t>(
        p.ekin, p.dir, rng, em, p.event, em::kElectronTrackingCut<real_t>(), real_t(1e-6));
    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = r.local_deposit; }
    if (!r.gamma_survives) { return false; }
    p.ekin = r.gamma_energy;
    p.dir = r.gamma_dir;
    return true;
  }

  if (proc == em::GammaProcess::kRayleigh && s.rayleigh != nullptr) {
    rep.process = ProcessId::fRayleigh;
    // Coherent: the photon is redirected, no energy transferred, nothing deposited.
    // The scattering element is chosen by Rayleigh cross section, as SelectRandomAtom does.
    const data::Material<real_t>& mm = s.materials[mat];
    const real_t target = rng.uniform() * xs.rayleigh;
    real_t acc = real_t(0);
    int z = static_cast<int>(mm.z[0] + real_t(0.5));
    for (int i = 0; i < mm.n_elements; ++i) {
      z = static_cast<int>(mm.z[i] + real_t(0.5));
      acc += mm.n_atoms[i] * data::rayleigh_xs_per_atom(*s.rayleigh, z, p.ekin);
      if (acc >= target) { break; }
    }
    p.dir = data::sample_rayleigh_direction(z, p.ekin, p.dir, rng);
    return true;
  }

  if (proc == em::GammaProcess::kPhotoelectric && s.photoelectric != nullptr) {
    rep.process = ProcessId::fPhotoelectric;
    // Pick the absorbing element weighted by its photoelectric cross section, as
    // G4VEmModel::SelectRandomAtom does.
    const data::Material<real_t>& mm = s.materials[mat];
    const real_t target = rng.uniform() * xs.photoelectric;
    real_t acc = real_t(0);
    int z = static_cast<int>(mm.z[0] + real_t(0.5));
    for (int i = 0; i < mm.n_elements; ++i) {
      z = static_cast<int>(mm.z[i] + real_t(0.5));
      acc += mm.n_atoms[i] * data::photoelectric_xs_per_atom(*s.photoelectric, z, p.ekin);
      if (acc >= target) { break; }
    }
    const auto r = data::sample_photoelectric(*s.photoelectric, z, p.ekin, p.dir, rng);
    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = r.local_deposit; }
    if (r.electron_produced) {
      em.push(ParticleType::kElectron, r.electron_dir, r.electron_ekin, p.event);
    }
    return false;  // photon absorbed
  }

  // Target atom chosen by pair cross section, as SelectTargetAtom does.
  {
    rep.process = ProcessId::fGammaConversion;
    const data::Material<real_t>& mmp = s.materials[mat];
    const real_t tgt = rng.uniform() * xs.pair;
    real_t accp = real_t(0);
    int zp = static_cast<int>(mmp.z[0] + real_t(0.5));
    for (int i = 0; i < mmp.n_elements; ++i) {
      zp = static_cast<int>(mmp.z[i] + real_t(0.5));
      accp += mmp.n_atoms[i] * em::pair_xs_per_atom(p.ekin, mmp.z[i]);
      if (accp >= tgt) { break; }
    }
    const auto pr = em::sample_pair_production<real_t>(p.ekin, p.dir, zp, rng,
                                                      s.materials[mat].radiation_length);
    if (!pr.ok) {
      if (s.geometry.volumes[p.volume].score_index >= 0) { edep = p.ekin; }
      return false;
    }
    em.push(ParticleType::kElectron, pr.electron_dir, pr.electron_ekin, p.event);
    em.push(ParticleType::kPositron, pr.positron_dir, pr.positron_ekin, p.event);
  }
  return false;  // photon consumed
}

/// Advances one electron or positron by a single step. A positron that falls below the
/// tracking cut emits its two annihilation photons before dying.
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_lepton(const Scene<real_t>& s, TrackState<real_t>& p, bool is_positron,
                                   Rng& rng, Emitter& em, real_t& edep, StepReport<real_t>& rep,
                                   vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  rep = StepReport<real_t>{};
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld) { return false; }

  // Termination guards. Without these a lepton can stop making progress near the end of
  // its range and never fall below the tracking cut - the drain loop then never empties.
  // In FP32 this is not hypothetical: it stalled every batch until these were added.
  //
  // The threshold is absolute and physically motivated rather than tied to the numeric
  // precision: 10 um of residual range is a few keV in these materials, two orders of
  // magnitude below Geant4's own 1 mm default production cut. Grinding below it is
  // pointless work - it cost ~32 of the 71 steps per event in the FP64 build.
  const real_t kMinUsefulRange = real_t(1e-2);  // mm
  const int mat0 = geom::material_at(s.geometry, p.volume, p.pos);
  rep.material = mat0;
  const bool no_range_left = (s.range_table->lookup(mat0, p.ekin) < kMinUsefulRange);

  const bool below_cut = (p.ekin < em::kElectronTrackingCut<real_t>()) || no_range_left;
  if (!below_cut) {
    const int mat = mat0;
    const real_t range = s.range_table->lookup(mat, p.ekin);

    int next_volume = geom::kOutsideWorld;
    const real_t d_boundary =
        geom::step_to_boundary(s.geometry, p.volume, p.pos, p.dir, next_volume);

    // Discrete delta-ray emission competes with the boundary and the step limits. Geant4
    // splits ionisation the same way: transfers below the production cut are the continuous
    // restricted dE/dx, transfers above it produce a transportable electron. These
    // interaction lengths are *true* path lengths, like every other limit below.
    const real_t delta_xs = em::delta_ray_xs(s.materials[mat], p.ekin, is_positron);
    const real_t d_delta = (delta_xs > real_t(0)) ? -log(rng.uniform()) / delta_xs
                                                 : geom::kInfinity<real_t>();
    // Bremsstrahlung photon emission is a third discrete competitor.
    const real_t brem_xs =
        (s.brems != nullptr && s.processes.bremsstrahlung)
            ? s.brems->xs_at(mat, is_positron, p.ekin) : real_t(0);
    const real_t d_brem = (brem_xs > real_t(0)) ? -log(rng.uniform()) / brem_xs
                                                : geom::kInfinity<real_t>();
    // A positron can also annihilate in flight (G4eplusAnnihilation), a fourth competitor.
    const real_t annih_xs =
        (is_positron && s.processes.annihilation)
            ? em::annihilation_xs(s.materials[mat], p.ekin) : real_t(0);
    const real_t d_annih = (annih_xs > real_t(0)) ? -log(rng.uniform()) / annih_xs
                                                  : geom::kInfinity<real_t>();
    // ---- G4CoulombScattering, a fifth discrete competitor, and ONLY above 100 MeV.
    //
    // THE ENERGY GATE IS THE PROCESS AND NOT A SHORTCUT. `G4EmStandardPhysics` fetches
    // `G4EmParameters::MscEnergyLimit()` once and hands it to the e+- single-scattering model
    // as SetMinKinEnergy, SetLowEnergyLimit AND SetActivationLowEnergyLimit
    // (em/coulomb_scattering.cuh's table), so an electron below 100 MeV has the process on its
    // manager and inactive. Above it, Urban msc stops and WentzelVI plus single scattering take
    // over - the split is in ENERGY, because in option0 MscThetaLimit is pi and the two cover
    // the same angular range.
    //
    // AND THE DRAW IS CONDITIONAL, for the reason `step_hadron`'s decay draw is: below 100 MeV
    // no uniform is consumed, so every existing electron and positron result is unmoved. That
    // is not a small claim for this repository - B1 is a 6 MeV gamma beam whose secondaries
    // never reach 100 MeV, so its dose cannot move at all, and the proton and alpha depth-dose
    // curves are checked to a fraction of a sigma. The port's MSC below 100 MeV is still Urban,
    // which is what Geant4 runs there.
    const real_t coul_xs =
        (s.processes.coulomb_scattering && p.ekin >= em::kMscEnergyLimit<real_t>())
            ? em::coulomb_xs_per_volume(s.materials[mat],
                                        is_positron ? ParticleType::kPositron
                                                    : ParticleType::kElectron,
                                        p.ekin, em::coulomb_secondary_cut(s.range_cut),
                                        real_t(-1), real_t(-1))
            : real_t(0);
    const real_t d_coul = (coul_xs > real_t(0)) ? -log(rng.uniform()) / coul_xs
                                                : geom::kInfinity<real_t>();

    // Continuous-loss step limit, verbatim from
    // G4VEnergyLossProcess::AlongStepGetPhysicalInteractionLength with the G4EmParameters
    // defaults dRoverRange = 0.2 and finalRange = 1 mm (cutAsFinalRange is off, so the
    // production cut does not enter).
    const real_t finR = em::kFinalRange<real_t>();
    const real_t max_step =
        (range > finR) ? range * em::kDRoverRange<real_t>()
                             + finR * (real_t(1) - em::kDRoverRange<real_t>())
                                   * (real_t(2) - finR / range)
                       : range;

    // Multiple-scattering step limit, and the true -> geometric conversion it defines.
    const em::UrbanCoeffs<real_t>& uc = s.msc->coeffs[mat];
    const real_t lambda0 = s.msc->lambda_at(mat, is_positron, p.ekin);
    const real_t safety = geom::compute_safety(s.geometry, p.volume, p.pos);
    rep.safety = safety;
    const real_t t_msc = em::urban_step_limit(uc, lambda0, p.ekin, range, safety, is_positron,
                                             rng, p.msc_tlimit, p.msc_tlimitmin);
    // With MSC off, the step is not limited by scattering and no deflection is applied. The
    // track then travels in a straight line, losing energy continuously - which is what a
    // "no multiple scattering" study means.
    const real_t t_msc_eff = s.processes.multiple_scattering ? t_msc : geom::kInfinity<real_t>();

    // The true path length this step would take if geometry did not interrupt it.
    real_t t_step = fmin(fmin(max_step, t_msc_eff), fmin(fmin(d_delta, d_brem), d_annih));
    t_step = fmin(fmin(t_step, d_coul), range);

    // lambda at the energy left after the whole true step, for the general geometric branch.
    real_t lambda1 = real_t(-1);
    {
      const real_t rfin = fmax(range - t_step, real_t(0.01) * range);
      const real_t e_rfin = s.range_table->energy_from_range(mat, rfin);
      if (e_rfin > real_t(0)) { lambda1 = s.msc->lambda_at(mat, is_positron, e_rfin); }
    }
    em::MscStep<real_t> msc_state;
    const real_t z_step = em::urban_geom_path(t_step, lambda0, range, lambda1, p.ekin,
                                              units::electron_mass_c2<real_t>(), msc_state);

    // Geometry acts on the *geometric* length; a boundary can cut the step short.
    const bool hits_boundary = (d_boundary < z_step);
    const real_t geom_step = hits_boundary ? d_boundary : z_step;

    // ...and the energy loss and scattering act on the true length that corresponds to it.
    const real_t step_len = em::urban_true_path(geom_step, t_step, msc_state);
    // The true path, not the chord: MSC deflects within the step, so the displacement
    // |pos_after - pos_before| is shorter than the distance the electron actually ran and a
    // LET computed from it would be biased high. This is what G4Step::GetStepLength returns.
    rep.true_length = step_len;

    // A discrete process fires only if it was the limiting true length and geometry did not
    // cut in first.
    const bool annihilates =
        !hits_boundary && (d_annih <= t_step) && (d_annih <= d_brem) && (d_annih <= d_delta);
    const bool emits_brem =
        !hits_boundary && !annihilates && (d_brem <= t_step) && (d_brem <= d_delta);
    const bool emits_delta = !hits_boundary && !annihilates && !emits_brem && (d_delta <= t_step);
    // Coulomb scattering is LAST in the tie order, which is the order `G4EmStandardPhysics`
    // registers the processes in: msc, eIoni, eBrem, (annihilation for e+), then
    // `G4CoulombScattering`. `G4SteppingManager::DefinePhysicalStepLength` keeps the smallest
    // with a strict `<`, so a tie goes to whichever process the manager holds first. A tie
    // between two continuous distributions has probability zero; what the order must not do is
    // fire two processes on one step.
    const bool coulomb_scatters = !hits_boundary && !annihilates && !emits_brem && !emits_delta
                                  && (d_coul <= t_step);

    // The same competition, read back out as G4StepPoint::GetProcessDefinedStep would report
    // it. When nothing discrete won, the step was defined along its length by whichever limit
    // was tightest - the continuous-loss limit or multiple scattering - which is exactly the
    // distinction Geant4 draws between fAlongStepDoItProc attributed to eIoni and to msc.
    if (hits_boundary) {
      rep.status = StepStatus::fGeomBoundary;
      rep.process = ProcessId::fTransportation;
    } else if (annihilates) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fAnnihilation;
    } else if (emits_brem) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fBremsstrahlung;
    } else if (emits_delta) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fIonisation;
    } else if (coulomb_scatters) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fCoulombScattering;
    } else {
      rep.status = StepStatus::fAlongStepDoItProc;
      rep.process = (t_msc_eff < max_step && t_msc_eff < range) ? ProcessId::fMultipleScattering
                                                                : ProcessId::fIonisation;
    }

    real_t e_after = s.range_table->energy_from_range(mat, range - step_len);
    // Second guard: the step must strictly reduce the energy. If the inverse range lookup
    // returns no decrease (rounding at small residual range), drop straight to zero rather
    // than requeueing a track that will make no progress next iteration either.
    if (e_after >= p.ekin) { e_after = real_t(0); }
    const real_t e_before = p.ekin;
    real_t loss = p.ekin - e_after;
    // The mean, kept because multiple scattering must not see the fluctuated value. Geant4
    // samples scattering in G4VMultipleScattering::AlongStepDoIt, which runs before the
    // ionisation process's AlongStepDoIt and derives its post-step energy from the range
    // table - the mean - not from whatever the fluctuation later drew.
    const real_t e_after_mean = e_after;

    // ---- energy-loss fluctuations, G4UniversalFluctuation.
    //
    // Applied to the *whole* continuous loss, not to the ionisation part of it, and that is
    // Geant4's structure rather than a simplification. G4LossTableManager builds one restricted
    // DEDX table per particle by summing every energy-loss process, hands it to the process it
    // has flagged `SetIonisation(true)` - G4eIonisation for a lepton - and every other loss
    // process returns from AlongStepDoIt immediately on `if(!isIonisation)`. So a single
    // AlongStepDoIt applies the combined ionisation-plus-sub-cut-radiative loss and fluctuates
    // it with the one model attached to G4eIonisation. The collision/radiative split below is
    // still applied afterwards, to decide how much is deposited here.
    //
    // The electron always takes the Glandz branch: sample_fluctuation's Gaussian branch
    // requires mass > electron_mass_c2, which is Geant4's own condition for it.
    if (loss < e_before && step_len > real_t(0)) {
      const ParticleDef<real_t> lpd = particle_def<real_t>(
          is_positron ? ParticleType::kPositron : ParticleType::kElectron);
      const real_t tmax = em::max_secondary_energy(e_before, is_positron);
      const real_t tcut = fmin(s.materials[mat].cut_electron, tmax);
      loss = em::sample_fluctuation(s.materials[mat], lpd, e_before, tcut, tmax, step_len, loss,
                                    real_t(1), rng);
      loss = fmin(fmax(loss, real_t(0)), e_before);
      e_after = e_before - loss;
    }

    // Split the continuous loss between collision (deposited here) and the restricted
    // radiative part. With the real Seltzer-Berger tables and a keV-scale gamma cut the
    // radiative share of the *continuous* loss is tiny - essentially all bremsstrahlung
    // leaves as explicit photons, emitted discretely below.
    const real_t col = em::collision_dedx(s.materials[mat], p.ekin, is_positron);
    const real_t rad = (s.brems != nullptr && s.processes.bremsstrahlung)
                           ? s.brems->dedx_at(mat, is_positron, p.ekin)
                                            : em::radiative_dedx(s.materials[mat], p.ekin);
    const real_t col_frac = (col + rad > real_t(0)) ? col / (col + rad) : real_t(1);

    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = loss * col_frac; }
    p.ekin = e_after;

    p.pos = p.pos + geom_step * p.dir;
    if (hits_boundary) {
      p.pos = p.pos + geom::kPushDistance<real_t>() * p.dir;
      p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
      p.msc_tlimit = real_t(0);  // stepStatus == fGeomBoundary: refresh rangeinit and fr
    }
    // Multiple scattering: deflection plus the correlated lateral displacement.
    // Geant4 skips the displacement when the step ends at the range limit, when it is
    // shorter than geomMin, or when the track is far enough from any boundary that a
    // sideways shift cannot change which volume it is in (the doverra test in
    // ComputeTruePathLengthLimit).
    const bool lat_disp = (step_len < range) && (step_len > em::kGeomMin<real_t>())
                          && !(range * uc.doverra < safety);
    const real_t e_scat =
        em::urban_scatter_energy(e_before, step_len, range, e_after_mean, col + rad);
    const auto msc_out = em::urban_sample_scattering(
        s.materials[mat], uc, lambda0, p.dir, step_len, geom_step, e_scat, e_before, lat_disp,
        is_positron, rng,
        (e_scat > real_t(0)) ? s.msc->lambda_at(mat, is_positron, e_scat) : real_t(-1));
    if (s.processes.multiple_scattering) { p.dir = msc_out.dir; }

    // Apply the displacement only as far as the post-step safety allows, exactly as
    // G4VMultipleScattering::AlongStepDoIt does: shift fully if it fits, scale it down to
    // the safety if it does not, and drop it if there is no room at all.
    {
      const Vec3<real_t>& d = msc_out.displacement;
      const real_t r2 = d.x * d.x + d.y * d.y + d.z * d.z;
      if (r2 > em::kGeomMin<real_t>() * em::kGeomMin<real_t>()) {
        const real_t disp_r = sqrt(r2);
        const real_t post_safety =
            real_t(0.99) * geom::compute_safety(s.geometry, p.volume, p.pos);
        if (disp_r <= post_safety) {
          p.pos = p.pos + d;
        } else if (post_safety > em::kGeomMin<real_t>()) {
          p.pos = p.pos + (post_safety / disp_r) * d;
        }
      }
    }

    // AFTER the displacement, and this is load-bearing rather than tidy.
    //
    // The step ends where MSC leaves the track, not where the straight-line advance put it.
    // Recorded before the displacement, every segment ended at a point the track then left
    // sideways, and the next step began from the displaced position - so consecutive segments
    // of one track did not share an endpoint and a charged trajectory drew as a row of
    // disconnected dashes with a gap of |displacement| between them. Visible in the viewer as
    // exactly that, and reported as such.
    //
    // The transport was never wrong: the displacement is applied to the track state and the
    // next step proceeds from there. It was the RECORD that drew a line to a place the track
    // did not end. Which is worse than cosmetic, because the viewer exists to make a transport
    // bug visible - docs/RISK.md R1, G5, G6 are all things a picture found - and a picture with
    // a built-in discontinuity is one nobody can read a real discontinuity out of.
    //
    // Geant4 draws a step as a straight line between its two step points, and the post-step
    // point is post-AlongStepDoIt, so this is also what Geant4 draws.
    traj.add(pos_before, p.pos,
             is_positron ? ParticleType::kPositron : ParticleType::kElectron, p.event,
             p.rng_key);

    // In-flight annihilation consumes the positron: two photons, no track survives.
    if (annihilates) {
      // If the step took the positron to rest, Geant4 takes the at-rest branch; the
      // in-flight sampler divides by tau and would produce a NaN direction at zero.
      const auto a = (p.ekin > real_t(0))
                         ? em::sample_annihilation_in_flight(p.ekin, p.dir, rng)
                         : em::sample_annihilation_at_rest<real_t>(rng);
      em.pos = p.pos;
      em.volume = p.volume;
      em.event = p.event;
      em.push(ParticleType::kGamma, a.dir1, a.energy1, p.event);
      em.push(ParticleType::kGamma, a.dir2, a.energy2, p.event);
      return false;
    }

    if (emits_brem && p.ekin > real_t(0) && s.sb != nullptr) {
      const data::Material<real_t>& mm = s.materials[mat];
      // Element choice weighted by Z^2 * n, the scaling of the brems cross section.
      real_t tot = real_t(0);
      for (int i = 0; i < mm.n_elements; ++i) { tot += mm.z[i] * mm.z[i] * mm.n_atoms[i]; }
      const real_t target = rng.uniform() * tot;
      real_t acc = real_t(0);
      int z = static_cast<int>(mm.z[0] + real_t(0.5));
      for (int i = 0; i < mm.n_elements; ++i) {
        z = static_cast<int>(mm.z[i] + real_t(0.5));
        acc += mm.z[i] * mm.z[i] * mm.n_atoms[i];
        if (acc >= target) { break; }
      }
      const int zi = s.sb->z_to_index[z];
      // Above 1 GeV G4eBremsstrahlung samples from G4eBremsstrahlungRelModel instead of the
      // Seltzer-Berger tables, which is also where those tables run out of data.
      const bool use_rel = (p.ekin > data::kSeltzerBergerLimit<real_t>());
      if (zi >= 0 || use_rel) {
        const real_t total_e = p.ekin + em::units_me<real_t>();
        const real_t dc = data::migdal_constant<real_t>() * mm.electron_density * total_e
                          * total_e;
        const real_t k =
            use_rel ? em::sample_rel_brem_energy(mm, z, p.ekin, mm.cut_gamma, p.ekin, rng)
                    : data::sample_brems_energy(s.sb->elements[zi], p.ekin, mm.cut_gamma,
                                                p.ekin, dc, is_positron, rng);
        if (k > real_t(0) && k < p.ekin) {
          // G4ModifiedTsai samples from the pre-emission electron kinetic energy.
          const auto gdir = data::sample_brems_direction(p.ekin, p.dir, rng);
          em.pos = p.pos;
          em.volume = p.volume;
          em.event = p.event;
          em.push(ParticleType::kGamma, gdir, k, p.event);
          p.ekin -= k;
        }
      }
    }

    if (emits_delta && p.ekin > real_t(0)) {
      const auto d = em::sample_delta_ray(s.materials[mat], p.ekin, p.dir, is_positron, rng);
      if (d.produced) {
        em.pos = p.pos;
        em.volume = p.volume;
        em.event = p.event;
        em.push(ParticleType::kElectron, d.delta_dir, d.delta_ekin, p.event);
        p.ekin = d.primary_ekin;
        p.dir = d.primary_dir;
      }
    }

    // ---- G4CoulombScattering::PostStepDoIt, on the POST-step state.
    //
    // After the continuous loss and after the multiple scattering, because that is the order
    // Geant4 runs them in: every AlongStepDoIt, `G4Step::UpdateTrack`, then the one PostStepDoIt
    // that won. So `p.ekin` here is `track.GetKineticEnergy()`, which is the energy
    // `G4VEmProcess::PostStepDoIt` reads and passes to SampleSecondaries.
    if (coulomb_scatters && p.ekin > real_t(0)) {
      const ParticleType lt =
          is_positron ? ParticleType::kPositron : ParticleType::kElectron;
      coulomb_apply(s, p, lt, mat, coul_xs, rng, em, edep, rep);
    }

    if (p.ekin >= em::kElectronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld) {
      return true;
    }
  }

  // Dying: deposit whatever is left, then annihilate if it is a positron.
  //
  // Reached both by a track that was already below its cut on entry and by one that fell below
  // it during the step, so the process that defined the step is left alone if the transport
  // above already determined one.
  rep.status = StepStatus::fStopAndKill;
  if (rep.process == ProcessId::fNotDefined) { rep.process = ProcessId::fBelowTrackingCut; }
  if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += p.ekin; }
  if (is_positron) {
    const auto a = em::sample_annihilation_at_rest<real_t>(rng);
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    em.push(ParticleType::kGamma, a.dir1, a.energy1, p.event);
    em.push(ParticleType::kGamma, a.dir2, a.energy2, p.event);
  }
  return false;
}

/// Advances one proton or alpha by a single step.
///
/// The same shape as step_lepton - limits compete, geometry cuts in, the continuous loss comes
/// out of the range table by inversion - but almost none of the same numbers. What is genuinely
/// different, rather than merely renamed:
///
///   * **Multiple scattering is TWO models, and which one is a property of the species.**
///     Geant4 prints its own answer at initialisation and it reads:
///
///         msc:  for proton  SubType= 10
///                 WentzelVIUni : Emin=    0 eV  Emax=  100 TeV
///         msc:  for alpha  SubType= 10
///                     UrbanMsc : Emin=    0 eV  Emax=  100 TeV
///
///     `G4EmBuilder::ConstructLightHadrons` calls `SetEmModel(new G4WentzelVIModel())` on the
///     `G4hMultipleScattering` it gives mu+-, pi+-, K+- and p/pbar; nothing calls it on the
///     ions', and `G4hMultipleScattering::InitialiseProcess` then defaults to
///     `new G4UrbanMscModel()`. So the table this function dispatches on is
///
///         mu+- pi+- K+- p pbar        WentzelVIUni      SetEmModel, ConstructLightHadrons
///         alpha He3 deuteron triton   UrbanMsc          no model set, the default
///         GenericIon                  UrbanMsc          the shared "ionmsc", no model set
///
///     and it is not written from the source alone: the `models` column of
///     `ref/oracle/species_processes.csv` is model 0's name off each species' own process
///     manager in a constructed QBBC - the same thing `/particle/process/dump` prints - and
///     `tests/test_species.cu` compares `uses_wentzel_msc` against it species by species.
///     `uses_wentzel_msc` in core/particle.cuh is where the table lives; because `type` is a
///     template parameter of `run_step_hadron` the branch here is resolved at compile time and
///     neither kernel carries the other model.
///
///     Until P14b that predicate named a SUBSTITUTION rather than a dispatch - every ion went
///     through WentzelVI, because this file's Urban had only the electron's stepping half
///     (`is_positron` in its step limit and its sampler, and an e-/e+ transport-mfp table).
///     em/urban_msc.cuh is now general across mass and charge and the substitution is gone.
///     What it was worth is in docs/RISK.md V61.
///
///     What differs between the two, and all of it is `G4EmParameters` picking on the
///     PARTICLE rather than on the model (`G4EmTableUtil::PrepareMscProcess` on
///     `GetPDGMass() > MeV`, `G4VMscModel::InitialiseParameters` on `abs(PDGEncoding) == 11`):
///     both get `facrange` 0.2 rather than the lepton's 0.04, both get the `fMinimal` step
///     limit type, and both have their lateral displacement switched off
///     (`MuHadLateralDisplacement` false), so a heavy particle is deflected but never shifted
///     sideways and the branch step_lepton spends twenty lines on does not exist here. A
///     comment that attributes 0.2 to WentzelVI and 0.04 to Urban - which this one used to -
///     has the mechanism backwards.
///
///     One thing Urban needs that WentzelVI does not: a step-limit state carried between
///     steps. fMinimal recomputes its limit only at a geometry boundary, so `msc_tlimit` on
///     the track holds it, in three states rather than two - see `em::kMscAtBoundary`.
///
///   * **Nuclear stopping is applied after the continuous loss**, not integrated into the range
///     table. G4NuclearStopping is a G4VEmProcess that acts along the step and builds no DEDX
///     table, so Geant4's range for a proton is its range against ionisation alone. Putting
///     nuclear stopping into the table put the proton range 25% short at 10 keV; leaving it out
///     of the *stepper* would lose the energy entirely. See hadron_total_dedx.
///
///   * **The loss is all local, and there is no discrete radiative process at all.** Two
///     separate statements, and the second is the gap.
///
///     The continuous half is genuinely nothing. `ref/oracle/hadron_radiative.csv`'s dedx_brem
///     and dedx_pair columns are zero for every proton and alpha row below a GeV, so there is
///     no share to split off - and for the muon, which this function now steps,
///     `muon_models.csv` puts the restricted radiative share of dE/dx in water at 5e-5 of the
///     total at 1.6 GeV, 3.6e-4 at 10 GeV and 2.7e-3 at 100 TeV. Below every tolerance here.
///
///     The DISCRETE half is missing and is not small at high energy. Every charged hadron in
///     QBBC carries `hBrems` and `hPairProd`, and mu+- carry `muBrems` and `muPairProd`
///     (`ref/oracle/species_processes.csv`), and nothing below samples one. em/muon_radiative.
///     cuh has both models' dE/dx and cross sections, exact, and neither model's
///     `SampleSecondaries` - so the interaction length could be drawn and the final state could
///     not be applied, which is why it is absent rather than half-wired. The mean free paths say
///     where it starts to matter: a mu- in water at 1 GeV has a 649 m brem and 1.8 km pair
///     length against a 6 m range, and at 10 GeV an 86 m pair length against a 58 m range - so
///     a muon above about 10 GeV is transported here without its dominant loss channel. For
///     pi+- pair production does not begin until 1.12 GeV and for the proton until 7.5 GeV.
///     Pre-existing rather than new: the proton has been stepped without these two processes for
///     as long as it has been stepped. docs/PORTED.md 1.3 and docs/RISK.md V38 carry the numbers
///     and the reason no energy refusal was added.
///
///   * **THE NUCLEAR INTERACTIONS ARE DECAY AND ELASTIC SCATTERING**, and single Coulomb
///     scattering beside them. This used to read "there are no nuclear interactions"; P8 added
///     `G4Decay` and P8b added `hadElastic` and `CoulombScat`. Four discrete competitors now:
///     the delta ray, the decay, `CoulombScat` and `hadElastic`, in the order the process
///     manager holds them.
///
///     `hadElastic` was P8's one named blocker and it was the TARGET DRAW rather than the
///     physics: `xs::store_sample_za` needs an `ElementIsotopes` per element and no table held
///     abundances. `data/isotope_abundance.hh` is G4NistElementBuilder's, compared isotope by
///     isotope against a QBBC-initialised G4NistManager, so the recoil is (Z, A)-resolved as
///     `G4ChipsElasticModel`'s per-isotope tables require rather than approximated by aeff[Z].
///     Which (cross section, model) pair per species is `had::elastic_channel`.
///
///     What QBBC still gives a charged hadron and this function does not do, each with a
///     package and a counter:
///
///       - the inelastic final state (P9-P11), the large hole: docs/RESULT.md puts it at 19%
///         of a 210 MeV proton's dose and 33% of an 840 MeV alpha's. Absent from the cross
///         section rather than present and refused, so `kChargedHadronInelastic` is
///         structurally zero.
///       - the at-rest capture of a stopped negative hadron (P12), refused by name and
///         counted with the rest mass it costs - see `had::HadronicRefusal`, and note that in
///         stage 1 it is Geant4 that has it switched off and both sides decay instead.
///       - the ANTIPROTON's elastic scattering, for the opposite reason to everything else
///         here: `G4AntiNuclElastic` was not started in P5 and `G4ComponentAntiNuclNuclearXS`
///         is refused by P2, so `elastic_channel` answers `kAntiNucleusRefused` and a pbar
///         draws no hadronic interaction length while every other charged hadron does.
///       - `hBrems` and `hPairProd`, above - the models' dE/dx is exact and neither has a
///         `SampleSecondaries`.
///
///     An elastic recoil heavier than an alpha maps to `kGenericIon`, which has no kernel, so
///     it is counted by name with its kinetic energy rather than transported. For a 200 MeV
///     proton in water that is the oxygen recoils above the 70 keV threshold.
template <typename real_t, typename Rng, typename Emitter>
__host__ __device__ inline bool step_hadron(const Scene<real_t>& s, TrackState<real_t>& p,
                                   ParticleType type, const had::HadronicWiring<real_t>& had,
                                   Rng& rng, Emitter& em, real_t& edep,
                                   StepReport<real_t>& rep,
                                   vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  rep = StepReport<real_t>{};
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld || s.hadron_range == nullptr) { return false; }

  // ---- WHICH PARTICLE THIS IS, and for `kGenericIon` that is two answers.
  //
  // `type` is the species: it selects the process, the model, the table row and the elastic
  // channel. `h.def` is the DEFINITION: the mass and the charge. For every species but a real
  // nucleus they are two views of one thing; for `kGenericIon` they are not, because
  // `particle_def(kGenericIon)` is G4GenericIon's 938.2723 MeV placeholder and the track is an
  // oxygen recoil. See `em::SteppedHadron`, which carries both plus the nuclide.
  const em::SteppedHadron<real_t> h =
      (type == ParticleType::kGenericIon)
          ? em::stepped_ion<real_t>(ion_z_of(p.ion_za), ion_a_of(p.ion_za))
          : em::stepped_hadron<real_t>(type);
  const ParticleDef<real_t> pd = h.def;

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
  rep.material = mat;
  const data::Material<real_t>& mm = s.materials[mat];

  // ---- a nucleus with no nuclide, or one outside AME2012, is refused by name.
  //
  // Two ways to get here and both are loud rather than fast. A `kGenericIon` track whose
  // `ion_za` is zero is a bug in whatever emitted it - `BufferEmitter::push_nucleus` is the only
  // thing that sets the field and `seed_track_slot` sets it to zero, so a primary ion would be
  // one (and `G4RunManager::CheckSpecies` refuses that by name before a run starts). A nuclide
  // outside `data::nuclear_mass`'s AME2012 table gives mass zero, which
  // `dynamic_particle_beta` reads as a photon: beta exactly 1, no stopping power, and a nucleus
  // that crosses the geometry depositing nothing. That is the shape of failure this project
  // keeps writing up, so it is a counter and a deposit instead.
  if (h.type == ParticleType::kGenericIon && !(pd.mass > real_t(0))) {
    had::book_refusal<real_t>(had.books, had::HadronicRefusal::kIonWithoutNuclide, p.ekin);
    rep.status = StepStatus::fStopAndKill;
    rep.process = ProcessId::fBelowTrackingCut;
    if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += p.ekin; }
    return false;
  }

  // ---- the range, THROUGH THE SCALING, and that is not a cosmetic change.
  //
  // `range_for`, `dedx_for` and `energy_from_range_for` are `G4VEnergyLossProcess`'s
  // base-particle scaling: a species with no table of its own reads its base particle's at
  // `E * massRatio` and rescales by `chargeSqRatio` (em/hadron_range.cuh's own header block).
  // This function used to call `lookup(sp, ...)`, `dedx_at(sp, ...)` and
  // `energy_from_range(sp, ...)` - the SPECIES-level entry points, which take an energy already
  // in the table's own terms and apply no scaling at all.
  //
  // For the ten species that own a table both ratios are exactly 1, so every number those ten
  // have ever produced is unchanged - which is why this went unnoticed. For the DEUTERON and
  // the TRITON, which P1 gave kernels, massRatio is `m_p/m_d = 0.500246` and `m_p/m_t = 0.334`,
  // and the transport was reading a PROTON's range and dE/dx at the deuteron's own kinetic
  // energy. `core/particle.cuh`'s `hadron_base_particle` warns about exactly this arithmetic
  // ("a deuteron's range would have come out as a proton's of the same kinetic energy, which is
  // a factor of about two") and the fix it describes was made in the table and not at the call
  // site. docs/RISK.md V57 has the measurement.
  const real_t range = s.hadron_range->range_for(mm, h, mat, p.ekin);

  // The same two termination guards step_lepton needs, for the same reason: without them a
  // track can stop making progress near the end of its range and never fall below the cut, and
  // the drain loop never empties. 10 um of residual range is a few tens of keV for a proton.
  const real_t kMinUsefulRange = real_t(1e-2);  // mm
  if (p.ekin >= em::kHadronTrackingCut<real_t>() && range >= kMinUsefulRange) {
    const real_t cut = mm.cut_electron;

    int next_volume = geom::kOutsideWorld;
    const real_t d_boundary =
        geom::step_to_boundary(s.geometry, p.volume, p.pos, p.dir, next_volume);

    // Discrete delta-ray production competes with the boundary and the step limits.
    //
    // REFUSED BY NAME FOR A REAL ION, AND THE REFUSAL IS UNREACHABLE RATHER THAN APPROXIMATE.
    //
    // `em::hadron_delta_xs` and `em::sample_hadron_delta` both derive the projectile's
    // definition from its SPECIES, and for `kGenericIon` that is G4GenericIon's placeholder - so
    // handing either of them an oxygen recoil would compute the delta-ray rate of a singly
    // charged 938 MeV particle. Threading `h.def` and `h.a` through them is a change in three
    // shared EM headers for a branch that cannot fire: an ion's transfer window opens only when
    // `tmax > cut`, and `tmax = 2 m_e b2g2 / (1 + 2 gamma m_e/M + (m_e/M)^2)`, so water's
    // 350 keV cut needs `beta^2 gamma^2 > 342` - above about 17 GeV per nucleon. The ions this
    // transport makes are elastic recoils of tens of MeV at most.
    //
    // So the window is tested with the ion's OWN definition, which is exact, and an ion that
    // opens it is counted rather than sampled. The count is per STEP and not per interaction,
    // which is the opposite of every other entry in that ledger: what is missing here is the
    // whole delta-ray channel of an ion above 17 GeV/u, on every step of it, rather than one
    // final state that could not be applied. `hadronic_refusal_name` says so.
    real_t delta_xs = real_t(0);
    if (h.is_real_ion()) {
      if (em::hadron_max_secondary_energy(pd, p.ekin) > cut) {
        had::book_refusal<real_t>(had.books, had::HadronicRefusal::kIonDeltaRay, p.ekin);
      }
    } else {
      delta_xs = em::hadron_delta_xs(mm, type, p.ekin, cut, real_t(1e30));
    }
    const real_t d_delta = (delta_xs > real_t(0)) ? -log(rng.uniform()) / delta_xs
                                                  : geom::kInfinity<real_t>();

    // ---- G4Decay in flight, a second discrete competitor.
    //
    // THE DRAW IS CONDITIONAL AND THAT IS LOAD-BEARING. `decays_in_flight` is false for every
    // stable species, so a proton, an alpha, a deuteron and a He3 consume exactly the uniforms
    // they consumed before P8 and their B1 doses are unmoved - which matters because the
    // proton's is this project's headline number and is checked to a fraction of a sigma. An
    // unconditional `-log(rand)` here would shift it, and the shift would look like physics.
    //
    // The triton is the interesting row: `G4Decay::IsApplicable` reads the LIFETIME, so the
    // triton gets the process (17.774 years >= 0), and `GetPDGStable()` is true, so it never
    // fires. `decays_in_flight` asks both questions in that order, as P4's file does.
    //
    // A REAL NUCLEUS NEVER DECAYS HERE, and that is QBBC rather than a gap. An unstable
    // nuclide's decay is `G4RadioactiveDecay`, which QBBC does not register at all -
    // docs/HADRONIC_PLAN.md section 2 lists the `RadioactiveDecay` dataset as present on this
    // machine and out of scope - and `G4Decay` is given to a species by `G4DecayPhysics`'s loop
    // over the particle table, which runs before any real ion exists. `decays_in_flight` would
    // be asked about G4GenericIon's own placeholder PDG code (1000000000, Z = A = 0), which is
    // not a nuclide and is in no decay table, so the answer would be right by accident; this
    // says it on purpose.
    const real_t d_decay =
        (had.decay && !h.is_real_ion() && had::decays_in_flight(type))
            ? had::decay_in_flight_length<real_t>(type, pd.mass, p.ekin, rng)
            : geom::kInfinity<real_t>();

    // ---- G4CoulombScattering in flight, a third discrete competitor.
    //
    // WHICH SPECIES, AND THE THRESHOLD THAT IS NOT AN ENERGY LIMIT.
    //
    // `has_coulomb_scattering` is the table in em/coulomb_scattering.cuh's header: mu+-, pi+-,
    // K+- and p/pbar get `G4CoulombScattering` from `G4EmBuilder::ConstructLightHadrons`
    // alongside their WentzelVI msc, and the ions - alpha, He3, deuteron, triton, GenericIon -
    // do not get it at all, because the only constructor that gives them one is
    // `ConstructIonEmPhysicsSS` and option0 never calls it. So a stepped alpha draws no uniform
    // here and its B1 dose is unmoved, which is checked to about a per cent.
    //
    // The lower edge is `coulomb_table_min_energy`: the process's own 100 eV floor raised by
    // `MinPrimaryEnergy(part, mat)`, which is `sqrt(q2Max*<A^-2/3>/2 + m^2) - m` and therefore
    // MATERIAL-dependent. In water that is 1.75 MeV for a proton and 0.283 MeV for a muon; in
    // lead 0.295 and 0.0477. It is the energy `SetStartFromNullFlag` puts the table's first
    // non-zero node at, so below it the cross section is a tabulated zero rather than a small
    // number - and a port that used a single constant instead would give a 200 MeV proton this
    // process over its whole track in lead and over only part of it in water, or the reverse.
    //
    // CONDITIONAL, like the decay draw above and for the same reason: a species or an energy
    // that has no process must consume no uniform, or every existing number for it moves.
    const bool has_coul = s.processes.coulomb_scattering && em::has_coulomb_scattering(type)
                          && p.ekin >= em::coulomb_table_min_energy<real_t>(type, pd.mass,
                                                                           mm.inv_a23);
    const real_t coul_xs =
        has_coul ? em::coulomb_xs_per_volume(mm, type, p.ekin,
                                             em::coulomb_secondary_cut(s.range_cut),
                                             real_t(-1), real_t(-1))
                 : real_t(0);
    const real_t d_coul = (coul_xs > real_t(0)) ? -log(rng.uniform()) / coul_xs
                                                : geom::kInfinity<real_t>();

    // ---- hadElastic, a fourth discrete competitor - and the first HADRONIC interaction this
    // stepper has ever drawn.
    //
    // `elastic_xs_per_volume` is `G4CrossSectionDataStore::ComputeCrossSection` for whichever
    // (cross section, model) pair `G4HadronElasticPhysics::ConstructProcess` gives this species
    // - see `hadronic/elastic_wiring.cuh` for the table and for the three rows of it that are
    // easy to guess wrong. It returns zero for a species with no such process (the leptons) and
    // for the antiproton, whose data set P2 refuses and whose high-energy model P5 did not
    // write; the antiproton's refusal is booked on the dying branch below rather than here,
    // because a cross section of zero is not an interaction that could not be applied.
    //
    // THE PARTIAL SUMS ARE KEPT, and that is the point of `MaterialXs` being a struct: the
    // target element is drawn from the CUMULATIVE array this call leaves behind, at this energy
    // and in this material, and `G4HadronicProcess::PostStepDoIt` relies on the same pairing.
    // A draw against a stale array picks an element by another energy's cross sections and
    // nothing complains.
    //
    // Conditional on the flag and on the species, so a proton run with elastic switched off and
    // every alpha, deuteron, triton and muon consume exactly the uniforms they consumed before.
    hadronic::xs::MaterialXs<real_t> el_mxs{};
    const real_t el_xs =
        had.hadron_elastic
            ? had::elastic_xs_per_volume<real_t>(had.elastic, mm, type, p.ekin, el_mxs)
            : real_t(0);
    const real_t d_elastic = (el_xs > real_t(0)) ? -log(rng.uniform()) / el_xs
                                                 : geom::kInfinity<real_t>();

    // Continuous-loss limit, G4VEnergyLossProcess::AlongStepGetPhysicalInteractionLength with
    // the mu/hadron step function (0.2, 0.1 mm).
    const real_t finR = em::kHadronFinalRange<real_t>();
    const real_t dRoR = em::kHadronDRoverRange<real_t>();
    const real_t max_step =
        (range > finR) ? range * dRoR + finR * (real_t(1) - dRoR) * (real_t(2) - finR / range)
                       : range;

    // ---- MULTIPLE SCATTERING: URBAN FOR THE IONS, WENTZELVI FOR EVERYTHING ELSE HERE.
    //
    // `uses_wentzel_msc` in core/particle.cuh carries the table and its provenance
    // (`G4EmBuilder::ConstructLightHadrons` calls SetEmModel on the light hadrons' and the
    // muons' `G4hMultipleScattering`; nothing calls it on the ions', and
    // `G4hMultipleScattering::InitialiseProcess` then defaults to `new G4UrbanMscModel()`).
    // Until P14b that predicate named a SUBSTITUTION - every ion was run through WentzelVI
    // because this file's Urban had only the electron's stepping half - and now it is a
    // dispatch.
    //
    // THE BRANCH COSTS NOTHING AT RUN TIME and that is why it is written as one. `type` is
    // `kType`, a template parameter of `run_step_hadron`, so `uses_wentzel_msc(type)` is a
    // compile-time constant in each of the thirteen instantiations: the proton kernel contains
    // no Urban code and the alpha kernel contains no WentzelVI code. Writing it as a run-time
    // branch instead would put both models' registers and both models' inlined tables into
    // every kernel, which docs/RISK.md V55 says is the thing that kills ptxas in this file.
    const bool urban_msc = !uses_wentzel_msc(type);

    const real_t safety = geom::compute_safety(s.geometry, p.volume, p.pos);
    rep.safety = safety;

    // WentzelVI's state, live only on the !urban_msc side.
    constexpr real_t kCosThetaLim = real_t(-1);  // G4EmParameters::MscThetaLimit() = pi
    em::WentzelMscState<real_t> st{};
    em::WentzelElementXs<real_t> els{};
    // Urban's, live only on the other. The coefficients are a function of the material's Zeff
    // alone, so the electron's table already holds the right ones; `s.msc` is null in a run
    // that builds no lepton tables (Scene's own contract: a null table is a process switched
    // off) and then they are computed here instead.
    const em::UrbanCoeffs<real_t> uc =
        (s.msc != nullptr) ? s.msc->coeffs[mat] : em::urban_coeffs(mm);
    real_t urban_lambda0 = real_t(0);
    em::MscStep<real_t> urban_state{};

    real_t t_msc = geom::kInfinity<real_t>();
    if (urban_msc) {
      // The transport mean free path, EVALUATED AND NOT LOOKED UP. No ion has a cross-section
      // table: `G4VMscModel::GetParticleChangeForMSC` builds one only for a particle under
      // 1 GeV not named GenericIon, and `SetForceBuildTable` is called nowhere in 11.1.1, so
      // `GetTransportMeanFreePath` falls through to `CrossSectionPerVolume` at the energy
      // asked for. See `em::urban_heavy_lambda`, which is also why it is `__noinline__`.
      //
      // The charge here is the BARE one from the definition - 2 for an alpha, Z for a recoil
      // ion - and not the effective charge. `G4UrbanMscModel::SetParticle` reads
      // `GetPDGCharge()/eplus` when the track starts and never refreshes it, unlike
      // `G4VEnergyLossProcess`, which refreshes `chargeSqRatio` from the model on every step
      // under `if(isIon)`. So the same ion scatters with Z and loses energy with q_eff(E).
      urban_lambda0 = em::urban_heavy_lambda(mm, p.ekin, pd.mass, pd.charge);
      if (s.processes.multiple_scattering) {
        t_msc = em::urban_step_limit_heavy(uc, urban_lambda0, em::kFacRangeMuHad<real_t>(),
                                           p.ekin, pd.mass, range, safety, max_step, rng,
                                           p.msc_tlimit);
      }
    } else {
      // No table here either, but for a different reason: WentzelVI's transport cross section
      // is closed-form, so lambda comes straight from wv_transport_xs. That call also fills
      // the per-element single-scattering tables the sampler needs to pick which atom a
      // discrete scatter happened on, which is why it is made even where only lambda is wanted.
      st.range = range;
      st.pre_kin_energy = p.ekin;
      st.eff_kin_energy = p.ekin;
      st.single_scattering_mode = false;
      {
        const auto s0 = em::wentzel_setup(pd, type, p.ekin, mm.inv_a23,
                                          static_cast<int>(mm.z[0] + real_t(0.5)), cut,
                                          kCosThetaLim);
        st.cos_tet_max_nuc = s0.cos_tet_max_nuc;
      }
      // Two different cross sections, and taking one for the other switches multiple scattering
      // off without saying so. wv_transport_xs called with cos_theta = 1 *returns* the transport
      // cross section above that angle, which is identically zero; what it is called for here is
      // its out-parameter xtsec, the total single-scattering rate, and the per-element table the
      // sampler picks an atom from. lambda_eff is the transport mean free path and comes from
      // wentzel_lambda, integrated over the whole angular range.
      //
      // The first version of this took the return value. lambda_eff was then 1e30, every step
      // fell into single-scattering mode with a zero rate, and wv_sample_scattering returned the
      // incoming direction unchanged: a proton beam that never scattered at all, with no error
      // and no warning anywhere.
      em::wv_transport_xs(mm, pd, type, p.ekin, cut, kCosThetaLim, real_t(1),
                          st.cos_tet_max_nuc, els, st.xtsec);
      st.lambda_eff = em::wentzel_lambda(mm, type, pd, p.ekin, cut, kCosThetaLim);
      if (s.processes.multiple_scattering) {
        t_msc = em::wv_step_limit(mm, pd, type, p.ekin, range, st.lambda_eff,
                                  st.cos_tet_max_nuc, kCosThetaLim, safety, s.range_cut,
                                  max_step, em::kHadronFacRange<real_t>());
      }
    }

    // The true path length this step would take if geometry did not interrupt it.
    real_t t_step = fmin(fmin(max_step, t_msc), fmin(d_delta, d_decay));
    t_step = fmin(fmin(t_step, fmin(d_coul, d_elastic)), range);

    // The energy after the whole true step, and the transport mfp at the mean energy - both
    // are inputs to the true/geometric conversion, so both are computed from the uninterrupted
    // step before geometry can cut it short.
    // `p.ekin` is the pre-step energy the charge-square ratio is frozen at, which is the
    // argument `energy_from_range_for` asks for and the same thing Geant4 freezes: see its
    // header, and G4VEnergyLossProcess, which sets chargeSqRatio in
    // AlongStepGetPhysicalInteractionLength and holds it for the whole step.
    const real_t e_end = s.hadron_range->energy_from_range_for(
        mm, h, mat, fmax(range - t_step, real_t(0)), p.ekin);
    real_t lambda_eff_end = st.lambda_eff;
    real_t cos_tet_max_end = st.cos_tet_max_nuc;
    real_t z_step = t_step;
    if (urban_msc) {
      // Urban's general true->geometric branch needs the mean free path at the energy left at
      // the RESIDUAL RANGE, not at the mean energy WentzelVI uses. Geant4's own two lines:
      //     rfin = max(currentRange - tPathLength, 0.01*currentRange)
      //     lambda1 = GetTransportMeanFreePath(particle, GetEnergy(particle, rfin, couple))
      // and the 1% floor is what keeps a step that consumes the whole range from asking for
      // the mean free path at zero energy.
      real_t lam_rfin = real_t(-1);
      if (s.processes.multiple_scattering) {
        const real_t rfin = fmax(range - t_step, real_t(0.01) * range);
        const real_t e_rfin = s.hadron_range->energy_from_range_for(mm, h, mat, rfin, p.ekin);
        if (e_rfin > real_t(0)) {
          lam_rfin = em::urban_heavy_lambda(mm, e_rfin, pd.mass, pd.charge);
        }
        z_step = em::urban_geom_path(t_step, urban_lambda0, range, lam_rfin, p.ekin, pd.mass,
                                     urban_state);
      }
    } else {
      const real_t e_mid = real_t(0.5) * (e_end + p.ekin);
      if (e_mid > real_t(0)) {
        em::WentzelElementXs<real_t> tmp{};
        real_t xt = real_t(0);
        const auto sm = em::wentzel_setup(pd, type, e_mid, mm.inv_a23,
                                          static_cast<int>(mm.z[0] + real_t(0.5)), cut,
                                          kCosThetaLim);
        cos_tet_max_end = sm.cos_tet_max_nuc;
        (void)em::wv_transport_xs(mm, pd, type, e_mid, cut, kCosThetaLim, real_t(1),
                                  cos_tet_max_end, tmp, xt);
        lambda_eff_end = em::wentzel_lambda(mm, type, pd, e_mid, cut, kCosThetaLim);
      }
      if (s.processes.multiple_scattering) {
        z_step = em::wv_geom_path(st, t_step, e_end, lambda_eff_end, cos_tet_max_end);
      }
    }

    const bool hits_boundary = (d_boundary < z_step);
    const real_t geom_step = hits_boundary ? d_boundary : z_step;

    real_t step_len = geom_step;
    if (s.processes.multiple_scattering) {
      if (urban_msc) {
        step_len = em::urban_true_path(geom_step, t_step, urban_state);
      } else {
        auto recompute = [&](real_t cos_min, real_t& xt) {
          return em::wv_transport_xs(mm, pd, type, st.eff_kin_energy, cut, kCosThetaLim,
                                     cos_min, st.cos_tet_max_nuc, els, xt);
        };
        step_len =
            em::wv_true_path(st, geom_step, e_end, lambda_eff_end, cos_tet_max_end, recompute);
      }
    }
    // See the note in step_lepton: the true path, which is what the energy loss below is
    // computed against and what G4Step::GetStepLength reports.
    rep.true_length = step_len;

    // A discrete process fires only if it was the limiting true length and geometry did not
    // cut in first. Decay is tested FIRST and strictly, so a tie goes to the delta ray - which
    // is arbitrary, exactly as Geant4's is: `G4SteppingManager::DefinePhysicalStepLength` keeps
    // the smallest with a strict `<`, so the winner of a tie is whichever process the process
    // manager holds first, which is the order the physics list registered them in. A tie between
    // two continuous distributions has probability zero; what it must not do is fire both.
    const bool decays = !hits_boundary && (d_decay <= t_step) && (d_decay < d_delta);
    const bool emits_delta = !hits_boundary && !decays && (d_delta <= t_step);
    // Coulomb scattering last, as in step_lepton: `G4EmBuilder::ConstructLightHadrons`
    // registers msc, hIoni, hBrems, hPairProd and then G4CoulombScattering, and G4DecayPhysics
    // runs before G4EmStandardPhysics in QBBC's constructor list - so Decay is ahead of every EM
    // process on the manager and CoulombScat is behind all of them. See the note on ties above.
    const bool coulomb_scatters =
        !hits_boundary && !decays && !emits_delta && (d_coul <= t_step);
    // And hadElastic after both, because `G4HadronElasticPhysicsXS` is registered after
    // `G4EmStandardPhysics` and `G4DecayPhysics` in QBBC's constructor list - so it is last on
    // the process manager and loses every tie. `ref/oracle/species_processes.csv` has the order
    // the constructed QBBC actually holds.
    const bool scatters_elastic = !hits_boundary && !decays && !emits_delta
                                  && !coulomb_scatters && (d_elastic <= t_step);

    // Read back out as G4StepPoint::GetProcessDefinedStep would report it. See the same block
    // in step_lepton.
    if (hits_boundary) {
      rep.status = StepStatus::fGeomBoundary;
      rep.process = ProcessId::fTransportation;
    } else if (decays) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fDecay;
    } else if (emits_delta) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fIonisation;
    } else if (coulomb_scatters) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fCoulombScattering;
    } else if (scatters_elastic) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fHadronElastic;
    } else {
      rep.status = StepStatus::fAlongStepDoItProc;
      rep.process = ProcessId::fIonisation;
    }

    // ---- continuous loss, verbatim from G4VEnergyLossProcess::AlongStepDoIt.
    //
    // Not the pure range inversion step_lepton uses. Geant4 takes dE/dx times the length and
    // only inverts the range when that product exceeds linLossLimit (0.01) of the energy,
    // and the difference is not academic:
    //
    //   * A range inversion needs a *tabulated* range on both sides. In a material with no
    //     stopping power - the vacuum a beam is launched through - the range is effectively
    //     infinite, the inverse lookup saturates at the top of the table, and the "energy went
    //     up, so the track must have stopped" guard below then kills the proton before it
    //     reaches the phantom. That is exactly what the first run of this stepper did: 2000
    //     protons, 100 MeV each, zero deposited.
    //   * For the short steps a boundary produces, dE/dx times length is also the more
    //     accurate of the two: it does not difference two large nearly-equal ranges.
    const real_t e_before = p.ekin;
    real_t e_after = real_t(0);
    real_t loss;
    // `G4EmParameters::LinearLossLimit` is 0.01 - and `G4ionIonisation`'s constructor calls
    // `SetLinearLossLimit(0.02)`, which overrides it for every species that process is
    // registered for. So the alpha, He3 and a generic ion invert the range table at twice the
    // fractional loss the proton does, and this was a flat 0.01 for all of them until the ion
    // needed the distinction. It changes which of the two expressions computes a step's loss,
    // not the loss - both are `G4VEnergyLossProcess::AlongStepDoIt`'s, and the long branch is
    // the accurate one - so it moves a stepped alpha's numbers slightly and cannot move a
    // total.
    const real_t kLinLossLimit =
        uses_ion_ionisation(type) ? real_t(0.02) : real_t(0.01);
    if (step_len >= range || e_before <= em::kHadronTrackingCut<real_t>()) {
      loss = e_before;
    } else {
      loss = step_len * s.hadron_range->dedx_for(mm, h, mat, e_before);
      if (loss > e_before * kLinLossLimit) {
        loss = e_before
               - s.hadron_range->energy_from_range_for(
                     mm, h, mat, fmax(range - step_len, real_t(0)), e_before);
      }
      loss = fmin(fmax(loss, real_t(0)), e_before);

      // ---- energy-loss fluctuations.
      //
      // The mean the stopping power gives is replaced by a sample around it. This is what
      // gives the Bragg peak its distal width: two protons that lose the mean at every step
      // stop at the same depth, and Geant4's do not. G4VEnergyLossProcess samples this on
      // every step of every charged particle unless the physics list asks for
      // fDummyFluctuation, so it is unconditional here too.
      //
      // *Which* model is a per-species decision and not a constant: G4EmBuilder registers
      // G4ionIonisation for alpha and He3 and G4hIonisation for everything else here, and the
      // two ask G4EmStandUtil::ModelOfFluctuations for different models. See
      // uses_ion_fluctuations in core/particle.cuh, which carries the whole table and the
      // reason it is a table. An alpha has never used the Universal model in a stock physics
      // list and a proton has never used the ion one; the two have the same mean, so choosing
      // wrongly costs nothing in a total dose and everything in a straggling width - which is
      // why this was wrong here for as long as it was.
      //
      // tcut is min(production cut, tmax), which is the same argument AlongStepDoIt passes.
      if (loss < e_before) {
        const real_t tmax = em::hadron_max_secondary_energy(pd, e_before);
        const real_t tcut = fmin(cut, tmax);
        if (uses_ion_fluctuations(type)) {
          // THE EFFECTIVE CHARGE SQUARED, and which species get one is `isIon`.
          //
          // `G4IonFluctuations` only ever sees an effective charge through
          // `SetParticleAndCharge`, which `G4VEnergyLossProcess::PostStepGetPhysicalInteraction
          // Length` calls under `if(isIon)` with the `q2` it just read from
          // `currentModel->ChargeSquareRatio(track)`. `G4EmTableUtil::CheckIon` leaves `isIon`
          // false for deuteron, triton, alpha+ and alpha by name, so an ALPHA keeps the bare 4
          // (`G4IonFluctuations::InitialiseMe` sets `effChargeSquare = charge*charge`) - and
          // He3 and every real nucleus do not, because neither is on that exclusion list.
          //
          // The comment this replaces said "a generic ion will need the real ratio here"; it
          // was also He3's, which was not transported when it was written.
          const real_t q2 =
              (h.is_real_ion() || em::uses_dynamic_effective_charge(type))
                  ? em::hadron_charge_sq_ratio<real_t>(mm, h, e_before)
                  : pd.charge * pd.charge;
          loss = em::sample_ion_fluctuation(mm, pd, e_before, tcut, tmax, step_len, loss, q2,
                                            rng);
        } else {
          loss = em::sample_fluctuation(mm, pd, e_before, tcut, tmax, step_len, loss,
                                        pd.charge * pd.charge, rng);
        }
        loss = fmin(fmax(loss, real_t(0)), e_before);
      }
      e_after = e_before - loss;
    }

    // ---- nuclear stopping, G4NuclearStopping::AlongStepDoIt.
    //
    // Applied to the energy *after* ionisation, evaluated at the mean of the pre- and post-step
    // energies, capped at the pre-step energy, and deposited locally. The guard is the model's
    // own and lives inside nuclear_stopping_dedx: above z1^2 MeV per nucleon it returns zero,
    // which is why a 100 MeV proton pays nothing for this call but a stopping one does.
    //
    // AND ONLY WHEN THE PHYSICS LIST REGISTERS THE PROCESS, WHICH QBBC DOES NOT.
    //
    // This was unconditional. `ref/oracle/species_processes.csv` lists every process on every
    // species' process manager in the constructed QBBC and there is no `nuclearStopping` row
    // for any of them: `G4EmParameters::maxNIELEnergy` initialises to 0.0, so
    // G4EmStandardPhysics never constructs the process to hand out. See
    // uses_nuclear_stopping in core/particle.cuh - which is now `false` for everything, and is
    // still a predicate because `/process/em/setMaxNIEL` turns it on and the source's
    // four-species split is then the right answer.
    if (uses_nuclear_stopping(type) && e_after > real_t(0) && step_len > real_t(0)) {
      const real_t t_mean = real_t(0.5) * (e_before + e_after);
      const real_t nl = fmin(step_len * em::nuclear_stopping_dedx(mm, type, t_mean), e_before);
      if (nl > real_t(0)) {
        e_after = fmax(e_after - nl, real_t(0));
        loss = e_before - e_after;
        // G4Step::GetNonIonizingEnergyDeposit. Part of edep, not additional to it.
        rep.non_ionizing = nl;
      }
    }

    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = loss; }
    p.ekin = e_after;

    p.pos = p.pos + geom_step * p.dir;
    if (hits_boundary) {
      p.pos = p.pos + geom::kPushDistance<real_t>() * p.dir;
      p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
      // fMinimal refreshes its step limit at a boundary and nowhere else - not even on the
      // first step of a track, which is what makes this three states rather than two. See
      // em::kMscAtBoundary. The field stays dead for a WentzelVI species, whose step limit
      // carries nothing between steps.
      if (urban_msc) { p.msc_tlimit = em::kMscAtBoundary<real_t>(); }
    }
    traj.add(pos_before, p.pos, type, p.event, p.rng_key);

    // ---- the scattering itself. lat_displacement is false for every heavy particle, so the
    // returned displacement is zero and there is nothing to apply - for both models, and it is
    // `G4EmParameters::MuHadLateralDisplacement` for both rather than a property of either.
    if (s.processes.multiple_scattering) {
      if (urban_msc) {
        // G4VMultipleScattering::AlongStepDoIt's own guard: a step that ran out the whole
        // range is not scattered, and neither is one below geomMin. The rest of the guards are
        // inside urban_sample_scattering, where Geant4 has them.
        if (step_len < range && step_len > em::kGeomMin<real_t>()) {
          // SampleScattering's first four lines choose the energy the ANGLE is sampled at, and
          // it is not the pre-step energy: over dtrl (5%) of the range it is the energy at the
          // residual range, over 1% it is the pre-step energy minus dE/dx times the step, and
          // below that it is the pre-step energy. `urban_scatter_energy` is those four lines;
          // the mean rather than the fluctuated loss reaches it, for the reason step_lepton
          // gives - Geant4 samples scattering in G4VMultipleScattering::AlongStepDoIt, which
          // runs before the energy-loss process's AlongStepDoIt.
          const real_t e_scat = em::urban_scatter_energy(
              e_before, step_len, range,
              s.hadron_range->energy_from_range_for(mm, h, mat,
                                                    fmax(range - step_len, real_t(0)), e_before),
              s.hadron_range->dedx_for(mm, h, mat, e_before));
          const real_t lam_scat = (e_scat > real_t(0))
                                      ? em::urban_heavy_lambda(mm, e_scat, pd.mass, pd.charge)
                                      : real_t(-1);
          const auto sc = em::urban_sample_scattering(
              mm, uc, urban_lambda0, p.dir, step_len, geom_step, e_scat, e_before,
              em::kHadronLateralDisplacement<real_t>(), pd.mass, pd.charge, false,
              em::kTlimitMinMinimal<real_t>(), rng, lam_scat);
          p.dir = sc.dir;
        }
      } else {
        const auto sc = em::wv_sample_scattering(mm, pd, type, st, els, cut, kCosThetaLim,
                                                 p.dir,
                                                 em::kHadronLateralDisplacement<real_t>(), rng);
        p.dir = sc.dir;
      }
    }

    if (emits_delta && p.ekin > real_t(0)) {
      const auto d = em::sample_hadron_delta<real_t>(type, p.ekin, cut, p.dir, rng);
      if (d.produced) {
        em.pos = p.pos;
        em.volume = p.volume;
        em.event = p.event;
        em.push(ParticleType::kElectron, d.delta_dir, d.delta_ekin, p.event);
        p.ekin = d.primary_ekin;
        p.dir = d.primary_dir;
      }
    }

    // ---- G4CoulombScattering::PostStepDoIt, on the POST-step state. See coulomb_apply.
    if (coulomb_scatters && p.ekin > real_t(0)) {
      coulomb_apply(s, p, type, mat, coul_xs, rng, em, edep, rep);
    }

    // ---- G4HadronElasticProcess::PostStepDoIt, on the POST-step state.
    //
    // After the continuous loss and the scattering, for the reason the decay branch gives: the
    // order is every AlongStepDoIt, then the one PostStepDoIt that won, so the elastic scatter
    // sees the energy and direction the step left the hadron with. The cross section it is
    // rejected against is the one the interaction length was drawn with - `el_xs`, at the
    // PRE-step energy - and `elastic_apply` recomputes it at this energy for the integral
    // approach, which for a pion or a proton is a real rejection (`fHadTwoPeaks`).
    if (scatters_elastic && p.ekin > real_t(0)) {
      const auto er = had::elastic_apply<real_t>(had.elastic, mm, type, p.ekin, p.dir,
                                                 s.range_cut, el_xs, el_mxs, rng);
      if (er.interacted) {
        p.dir = er.dir;
        p.ekin = er.energy;
        // Both deposits, with the same value: `G4HadronElasticProcess::PostStepDoIt` proposes
        // the sub-threshold recoil as LOCAL and as NON-IONIZING (note 5 of
        // elastic/elastic_process.cuh), which is what makes elastic scattering contribute to
        // NIEL and not to dose in a scorer that separates them.
        if (er.edep > real_t(0)) {
          if (s.geometry.volumes[p.volume].score_index >= 0) { edep += er.edep; }
          rep.non_ionizing += er.edep;
        }
        if (er.emit_recoil) {
          em.pos = p.pos;
          em.volume = p.volume;
          em.event = p.event;
          // TRANSPORTED, as of P8c. Hydrogen recoils as a proton and the five light nuclei as
          // themselves; everything heavier is `kGenericIon` carrying its own (Z, A), which is
          // what `push_nucleus` is for and what `TrackState::ion_za` holds. This used to read
          // "which has no kernel, so BufferEmitter::push counts it by name with its kinetic
          // energy" - a 200 MeV proton in water lost its oxygen recoils to that ledger, and
          // `build_all.bat`'s depth-dose gate failed on the 515 MeV of them in 6000 protons.
          em.push_nucleus(er.recoil_z, er.recoil_a, er.recoil_dir, er.recoil_ekin, p.event);
        }
        if (er.dropped_secondaries > 0) {
          had::book_refusal<real_t>(had.books, had::HadronicRefusal::kElasticDropped,
                                    er.recoil_ekin);
        }
      }
      if (!er.primary_survives) {
        // `efinal == 0` with no at-rest process: G4HadronElasticProcess proposes fStopAndKill.
        // Fall through to the dying branch, which deposits what is left and asks the at-rest
        // question - the same path a track that ran out of range takes.
        p.ekin = real_t(0);
      }
    }

    // ---- the decay itself, G4Decay::DecayIt on the POST-step state.
    //
    // After the continuous loss and after the scattering, because that is the order Geant4
    // runs them in - AlongStepDoIt for every process, then the one PostStepDoIt that won - so
    // the decay sees the energy and direction the step left the particle with. It deposits
    // nothing: `energyDeposit` in DecayIt is zero on the in-flight branch, and the parent's
    // whole four-momentum goes into the products.
    if (decays) {
      const real_t dir3[3] = {p.dir.x, p.dir.y, p.dir.z};
      decay::DecayProducts<real_t> products;
      decay::sample_decay<real_t>(pdg_code(type), pd.mass, p.ekin, dir3, false, rng, products);
      em.pos = p.pos;
      em.volume = p.volume;
      em.event = p.event;
      if (products.status == decay::DecayStatus::kOK) {
        had::emit_decay_products<real_t>(products, em, had.books);
      } else if (products.status != decay::DecayStatus::kStable) {
        // G4Decay's DECAY101 path: the parent is killed with no secondaries and a zero deposit.
        // kStable is NOT that path and is not a refusal - it is DecayIt returning an untouched
        // particle change, a no-op rather than an error, and P4's DecayStatus says so. It is
        // also unreachable from here, because `decays_in_flight` asked the same question before
        // the length was drawn. Every other status means P4 refused the channel, which cannot
        // happen for the species this stepper transports - so it is counted rather than trusted.
        had::book_refusal<real_t>(had.books, had::HadronicRefusal::kDecayChannel, p.ekin);
      }
      // The step report was already written in the read-back block above -
      // `(fPostStepDoItProc, fDecay)` - and it is NOT rewritten here. This branch used to set
      // `fStopAndKill`, which is wrong twice over. `core/step_report.cuh` reserves
      // `fStopAndKill` for the tracking-cut death and pairs it with `fBelowTrackingCut`, and
      // the port's own at-rest decay below reports `(fStopAndKill, fDecay)` because
      // G4StepStatus has no `fAtRestDoItProc` equivalent here - so an in-flight decay marked
      // `fStopAndKill` was indistinguishable from an at-rest one, and a stepping action
      // counting stopped pions would have counted every pion that decayed in flight as well.
      // `step_lepton`'s in-flight annihilation - the same shape of process, a PostStepDoIt that
      // consumes its primary - leaves `fPostStepDoItProc` standing for exactly this reason.
      return false;
    }

    if (p.ekin >= em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld) {
      return true;
    }
  }

  // Dying: whatever is left is deposited here.
  //
  // WHICH DEATH THIS IS matters, because only one of the two has an at-rest process. A track
  // that crossed the world boundary is gone with its energy; a track that ran out of range
  // STOPPED, and a stopped unstable particle is what `G4Decay`'s at-rest branch and P12's
  // stopping processes compete for. The port's stop is at `kHadronTrackingCut` or 10 um of
  // residual range rather than at exactly zero energy, so the residual kinetic energy is
  // deposited - which is also `G4Decay::DecayIt`'s own `energyDeposit` on the at-rest branch
  // (the parent is at rest by definition, so it is normally zero).
  const bool stopped_in_world = (p.volume >= 0) && (p.volume != geom::kOutsideWorld);
  rep.status = StepStatus::fStopAndKill;
  if (rep.process == ProcessId::fNotDefined) { rep.process = ProcessId::fBelowTrackingCut; }
  if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += p.ekin; }

  // `!h.is_real_ion()` for the reason the in-flight draw gives: a stopped nuclide's decay is
  // G4RadioactiveDecay's and QBBC registers none.
  if (stopped_in_world && had.decay && !h.is_real_ion()
      && had::decay_at_rest_allowed(type, had.stage)) {
    // The at-rest branch does NOT boost: the products are built in the parent's rest frame and
    // stay there. See P4's `sample_decay`, whose `at_rest` argument is exactly this.
    const real_t dir3[3] = {p.dir.x, p.dir.y, p.dir.z};
    decay::DecayProducts<real_t> products;
    decay::sample_decay<real_t>(pdg_code(type), pd.mass, p.ekin, dir3, true, rng, products);
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    if (products.status == decay::DecayStatus::kOK) {
      had::emit_decay_products<real_t>(products, em, had.books);
      rep.process = ProcessId::fDecay;
    } else if (products.status != decay::DecayStatus::kStable) {
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kDecayChannel, p.ekin);
    }
  } else if (stopped_in_world && had.stage == had::HadronicStage::kFinal) {
    // No decay at rest, in the FINAL stage: a stopping process pre-empted it and this port does
    // not have it. A hole worth the whole rest mass, so it is booked by name.
    //
    // THE STAGE CONDITION IS NOT DECORATION. In stage 1 the three at-rest captures are
    // inactivated on the Geant4 side, so a stopped pi-, K- or mu- decays there and here -
    // `decay_at_rest_allowed` took that branch - and a stopped ANTIPROTON does nothing on
    // either side, because it is stable (so `G4Decay::IsApplicable` is false for it) and its
    // only at-rest process is the one that was switched off. Booking a refusal for it in
    // stage 1 would count agreement as a gap and put a number in the report that says the
    // answer is missing energy it is not missing.
    const had::HadronicRefusal r = had::stopped_refusal(type);
    if (r != had::HadronicRefusal::kNumHadronicRefusals) {
      had::book_refusal<real_t>(had.books, r,
                                had::stopped_refusal_energy<real_t>(type, p.ekin));
    }
  }
  return false;
}

/// Advances one neutral hadron - a neutron or a pi0 - by a single step.
///
/// A third of the length of step_hadron, and the two thirds that are missing are missing for a
/// reason rather than for want of writing them. A neutral particle has no ionisation process,
/// so there is no continuous loss, no range table, no step function over the residual range,
/// no delta rays and no fluctuation; and `G4hMultipleScattering` is registered by charge, so
/// there is no multiple scattering and therefore no true-to-geometric path conversion. What is
/// left is what a G4Transportation-only track does: go straight until either geometry or one
/// discrete process stops it.
///
/// THE ONE DISCRETE PROCESS, and why it is one rather than three. `ref/oracle/hadronic_params.
/// csv` says `EnableNeutronGeneralProcess = 1`, and `ref/oracle/neutron_processes.csv` - dumped
/// from the real QBBC - shows the neutron carrying `Transportation`, `Decay` and
/// `NeutronGeneralProc` and nothing else. Elastic, inelastic and capture are sub-processes
/// inside that one process, competing through a single summed interaction length with the
/// winner chosen afterwards from cumulative partials. See physics/hadronic/neutron_general_xs.
/// cuh, which holds the grid, the contract and the socket P8 fills.
///
/// THE TIME CUT COMES FIRST, and it is not the tracking cut it is usually described as.
/// `G4NeutronGeneralProcess::PostStepGetPhysicalInteractionLength` opens with
///
///     if(track.GetGlobalTime() >= fTimeLimit) { fLambda = 0.0; return 0.0; }
///
/// - a zero interaction length, tested before the cross section is even looked up, so it beats
/// geometry and every other limit. `PostStepDoIt` then does
///
///     if(0.0 == fLambda) { theTotalResult->Initialize(track);
///                          theTotalResult->ProposeTrackStatus(fStopAndKill); return ...; }
///
/// and `Initialize` runs `InitializeLocalEnergyDeposit()`, which sets `theLocalEnergyDeposit`
/// and `theNonIonizingEnergyDeposit` to zero. **So the neutron's kinetic energy is not
/// deposited. It is discarded.** `G4NeutronKiller::PostStepDoIt` does the identical two lines
/// for the branch where the general process is off. That is worth stating flatly because the
/// natural assumption is the opposite - every other way a track dies in this stepper deposits
/// what is left where it stood - and depositing it would put energy into a phantom that Geant4
/// puts nowhere. The process exists "to kill particles to save CPU" (G4NeutronKiller.cc's own
/// description); it is a budget, not physics, and it does not conserve energy. The engine books
/// the discarded energy per event so that a run can say how much it lost this way rather than
/// leaving it to be discovered as a shortfall.
///
/// The energy half of the cut is inert: `kineticEnergyLimit` is 0.0 and the test is
/// `GetKineticEnergy() < kinEnergyThreshold`. See kNeutronEnergyLimit.
///
/// THE STAGE DECIDES HOW MANY DISCRETE PROCESSES THERE ARE, and for the neutron that is not a
/// subset relation (P8d). `had::HadronicStage`'s own comment has the source; the consequence
/// here is two different competitions:
///
///   kStage1   `hadElastic` and `nCapture` are separate processes on the neutron's manager -
///             which is what Geant4 builds when `EnableNeutronGeneralProcess` is false - so each
///             evaluates its OWN data store per step and draws its OWN interaction length, and
///             the smallest of the two (and of the decay length, and of the boundary) wins.
///             `neutronInelastic` is inactivated on the reference side and absent here.
///   kFinal    one interaction length from `G4NeutronGeneralProcess`'s combined table, then the
///             sub-process from the cumulative partials on the same grid. `inelastic` is
///             selectable and refused by name.
///
/// The two are not the same arithmetic applied twice: the general process's table is a 401-node
/// linear interpolation of the summed cross section at ITS node energies, and the per-process
/// path evaluates each G4PARTICLEXS data set at the track's own energy. Both are what Geant4
/// reads in the configuration they belong to (docs/PORTED.md 4.3), so both are here.
///
/// @param xs the combined table, or null. Null is the state a run with no `G4PARTICLEXSDATA`
///        is in: the cross section is then zero, `s_int` is infinite, and the neutron streams to
///        the world boundary - which is what a Geant4 neutron does with `NeutronGeneralProc`
///        inactivated.
/// `__host__ __device__` SINCE P8d, for the reason P8b gave when it did the same to
/// `step_hadron`: a wiring this size needs a host/device comparison, and the only device-only
/// things in it were `had::book_refusal`'s atomics and `vis::TrajectoryBuffer::add`'s cursor,
/// both of which already have host arms. `tests/test_neutron_general.cu` runs the same source
/// on both sides and compares every field; without it the only check on the sub-process branch
/// would be a whole B1 run, where a wiring defect is a dose a few per cent out.
template <typename real_t, typename Rng, typename Emitter>
__host__ __device__ inline bool step_neutral(const Scene<real_t>& s, TrackState<real_t>& p,
                                    ParticleType type,
                                    const had::NeutronGeneralXs<real_t>* xs,
                                    const had::HadronicWiring<real_t>& had, Rng& rng,
                                    Emitter& em, real_t& edep, StepReport<real_t>& rep,
                                    vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  rep = StepReport<real_t>{};
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld) { return false; }

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
  rep.material = mat;

  // ---- the neutron time cut. Before geometry, before the cross section, as above.
  //
  // Applied to the neutron only. G4NeutronKiller::IsApplicable is
  // `particle.GetParticleName() == "neutron"` and G4NeutronGeneralProcess exists only for the
  // neutron, so a pi0 has no time cut - it has a 8.5e-8 ns lifetime and decays long before any
  // clock matters, which is P4's business rather than this function's.
  if (type == ParticleType::kNeutron
      && p.global_time >= had::kNeutronTimeLimit<real_t>()) {
    rep.status = StepStatus::fStopAndKill;
    rep.process = ProcessId::fNeutronKiller;
    // No deposit. See the note above: this is the one death in this stepper that does not
    // hand its energy to the volume it happened in, and `edep` stays zero deliberately.
    return false;
  }

  // ---- geometry against the one discrete interaction length.
  int next_volume = geom::kOutsideWorld;
  const real_t d_boundary =
      geom::step_to_boundary(s.geometry, p.volume, p.pos, p.dir, next_volume);

  // ---- G4Decay in flight, competing with the hadronic processes and with geometry.
  //
  // DRAWN BEFORE THEM, which is both the process-manager order (Transportation, Decay,
  // hadElastic, neutronInelastic, nCapture, nKiller - see `ref/b1hadron/stage1_neutron.mac`'s
  // printed dump) and the order `step_hadron` uses for the same two processes. The port re-draws
  // every interaction length every step, so its stream differs from Geant4's by construction
  // either way; what this buys is that the two steppers agree with each other, which is what a
  // reader comparing them will assume.
  //
  // Both neutral hadrons have it and they are at opposite extremes. A free neutron's proper
  // lifetime is 880 s, so `beta*gamma*c*tau` is 2.6e11 mm at 100 MeV - the process is present
  // and inert, which is why the stage-1 macros leave Decay ACTIVE on the Geant4 side rather than
  // inactivating something that cannot fire. A pi0's is 8.5e-8 ns, so `c*tau` is 2.55e-5 mm and
  // `p/m * c*tau` at 100 MeV is about 3e-5 mm: it decays inside the first step, always, wherever
  // it was made. "pi0 decays at once" is that number and not a special case - the same
  // competition produces it.
  //
  // Conditional, for the reason `step_hadron` gives: a species that does not decay must draw
  // no uniform, or every existing result for it moves.
  const real_t d_decay =
      (had.decay && had::decays_in_flight(type))
          ? had::decay_in_flight_length<real_t>(type, particle_def<real_t>(type).mass, p.ekin,
                                                rng)
          : geom::kInfinity<real_t>();

  // Zero for a pi0, and for a neutron whose tables were never uploaded.
  //
  // ONE LOGARITHM OF THE ENERGY, taken here and handed to every lookup below, as
  // G4NeutronGeneralProcess takes one `fLogEnergy` per step and reads it from
  // ComputeGeneralLambda and GetProbability alike - and as every G4PARTICLEXS data set takes
  // `G4DynamicParticle::GetLogKineticEnergy()`. Only taken when there is a table to look up in:
  // `log` of a kinetic energy is defined for every track this function sees, but a transcendental
  // per step for a pi0 that has no table is a cost with no answer attached.
  const bool is_neutron = (type == ParticleType::kNeutron);
  const bool final_stage = (had.stage == had::HadronicStage::kFinal);
  // BOTH FLAGS, AND BOTH TABLES. `G4NeutronGeneralProcess` is ONE process holding three
  // sub-processes, so `/process/inactivate` takes all of them or none (docs/RISK.md V53) and
  // there is no Geant4 configuration in which the general process runs with its elastic
  // sub-process off. Requiring both flags says that rather than offering a configuration whose
  // reference does not exist. And both data sets, because the sub-process branch below
  // dereferences them - which is the state `Upload` refuses to let a table arrive without.
  const bool has_general_process =
      (xs != nullptr && is_neutron && final_stage && had.hadron_elastic && had.neutron_capture
       && had.neutron.elastic != nullptr && had.neutron.capture != nullptr);
  // In stage 1 the two sub-processes are separate processes with separate tables, so the gate is
  // per process rather than one gate for the pair - `/process/inactivate nCapture` really does
  // take capture off on its own there, which it cannot do in the final stage.
  const bool has_stage1_elastic = (is_neutron && !final_stage && had.hadron_elastic
                                   && had.neutron.elastic != nullptr);
  const bool has_stage1_capture = (is_neutron && !final_stage && had.neutron_capture
                                   && had.neutron.capture != nullptr);
  const real_t loge = (has_general_process || has_stage1_elastic || has_stage1_capture)
                          ? log(p.ekin)
                          : real_t(0);

  // The combined table's interaction length, in the final stage only.
  const real_t sigma = has_general_process ? xs->total(mat, p.ekin, loge) : real_t(0);
  const real_t s_int =
      (sigma > real_t(0)) ? -log(rng.uniform()) / sigma : geom::kInfinity<real_t>();

  // ---- stage 1: two processes, two data stores, two interaction lengths.
  //
  // The partial sums each `ComputeCrossSection` leaves behind are what that process's own
  // `SampleZandA` reads, and they are computed at the PRE-step energy - which for a neutral
  // particle is also the post-step energy, since there is no continuous loss. So the `MaterialXs`
  // filled here is the one the final state uses, and nothing recomputes it: `fXSType` is
  // `fHadNoIntegral` for a neutron (`G4HadronicProcess::BuildPhysicsTable` guards the integral
  // approach with `charge != 0.0`), so its PostStepDoIt skips the recompute entirely.
  //
  // The order is the neutron's process-manager order with the general process off - hadElastic
  // before nCapture - because the order is the random stream.
  hadronic::xs::MaterialXs<real_t> mxs_el{}, mxs_cap{};
  const real_t sigma_el =
      has_stage1_elastic
          ? had::neutron_sub_xs_per_volume<real_t>(had.neutron.elastic, s.materials[mat],
                                                   p.ekin, loge, mxs_el)
          : real_t(0);
  const real_t s_el =
      (sigma_el > real_t(0)) ? -log(rng.uniform()) / sigma_el : geom::kInfinity<real_t>();
  const real_t sigma_cap =
      has_stage1_capture
          ? had::neutron_sub_xs_per_volume<real_t>(had.neutron.capture, s.materials[mat],
                                                   p.ekin, loge, mxs_cap)
          : real_t(0);
  const real_t s_cap =
      (sigma_cap > real_t(0)) ? -log(rng.uniform()) / sigma_cap : geom::kInfinity<real_t>();

  // The shortest of the discrete lengths, and which one it was. `G4SteppingManager` asks every
  // process for a length and keeps the smallest; with the general process there is one hadronic
  // length and in stage 1 there are two, so this is the one place that has to know the stage.
  //
  // STRICTLY LESS THAN, in the order hadElastic-then-nCapture, so a tie goes to the process the
  // neutron's manager holds first - which is what a `<` loop over the manager does.
  real_t s_hadronic = s_int;
  had::NeutronSubProcess sub = had::NeutronSubProcess::kElastic;
  bool sub_from_general = has_general_process;
  if (!final_stage) {
    s_hadronic = geom::kInfinity<real_t>();
    if (s_el < s_hadronic) {
      s_hadronic = s_el;
      sub = had::NeutronSubProcess::kElastic;
    }
    if (s_cap < s_hadronic) {
      s_hadronic = s_cap;
      sub = had::NeutronSubProcess::kCapture;
    }
  }

  if (d_decay < s_hadronic && d_decay < d_boundary) {
    // Decay wins. The step ends where it fired, the parent is killed with no deposit, and the
    // products carry the whole four-momentum.
    rep.true_length = d_decay;
    rep.status = StepStatus::fPostStepDoItProc;
    rep.process = ProcessId::fDecay;
    p.pos = p.pos + d_decay * p.dir;
    traj.add(pos_before, p.pos, type, p.event, p.rng_key);
    const real_t dir3[3] = {p.dir.x, p.dir.y, p.dir.z};
    decay::DecayProducts<real_t> products;
    decay::sample_decay<real_t>(pdg_code(type), particle_def<real_t>(type).mass, p.ekin, dir3,
                                false, rng, products);
    em.pos = p.pos;
    em.volume = p.volume;
    em.event = p.event;
    if (products.status == decay::DecayStatus::kOK) {
      had::emit_decay_products<real_t>(products, em, had.books);
    } else if (products.status != decay::DecayStatus::kStable) {
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kDecayChannel, p.ekin);
    }
    return false;
  }

  if (s_hadronic >= d_boundary) {
    // Streaming. One step, requeued unless it left the world - the same shape as step_gamma's
    // boundary branch, including the push past the surface.
    rep.true_length = d_boundary + geom::kPushDistance<real_t>();
    rep.status = StepStatus::fGeomBoundary;
    rep.process = ProcessId::fTransportation;
    p.pos = p.pos + (d_boundary + geom::kPushDistance<real_t>()) * p.dir;
    traj.add(pos_before, p.pos, type, p.event, p.rng_key);
    p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
    return p.volume != geom::kOutsideWorld;
  }

  // ---- a sub-process fired. P8d; this branch reported `fNotDefined` until it did.
  //
  // The step ends at the interaction point first, because every final state below is written
  // there and because a track that does not survive has already moved.
  rep.true_length = s_hadronic;
  p.pos = p.pos + s_hadronic * p.dir;
  traj.add(pos_before, p.pos, type, p.event, p.rng_key);
  em.pos = p.pos;
  em.volume = p.volume;
  em.event = p.event;
  const bool scores = (s.geometry.volumes[p.volume].score_index >= 0);

  // In the final stage the sub-process comes from the general table's cumulative partials, and
  // it costs ONE uniform - `G4double q = G4UniformRand()` is the second statement of
  // `G4NeutronGeneralProcess::PostStepDoIt`. In stage 1 the competition already named it and no
  // uniform is drawn, which is what having two processes means.
  if (sub_from_general) { sub = xs->select(mat, p.ekin, loge, rng.uniform()); }

  if (sub == had::NeutronSubProcess::kInelastic) {
    // REFUSED BY NAME, with the energy it costs the answer. P9-P11 own
    // `G4BinaryCascade`/Bertini/FTFP; until one of them lands there is no final state to apply,
    // and Geant4 would have replaced this neutron with a shower of nucleons and fragments.
    //
    // The neutron is killed with its kinetic energy deposited locally. That is the conservative
    // disposal this file uses for every refusal (`kIonWithoutNuclide`, `kDecayChannel`) and it
    // is NOT what Geant4 does: an inelastic reaction spreads the energy over secondaries that
    // leave the volume. So a `kFinal` dose is wrong by this counter's energy, and the counter
    // exists so the size of that is read off a run rather than inferred from a disagreement.
    // Unreachable in `kStage1`, where the reference has `neutronInelastic` inactivated.
    had::book_refusal<real_t>(had.books, had::HadronicRefusal::kNeutronInelastic, p.ekin);
    rep.status = StepStatus::fStopAndKill;
    rep.process = ProcessId::fHadronInelastic;
    if (scores) { edep = p.ekin; }
    return false;
  }

  if (sub == had::NeutronSubProcess::kElastic) {
    rep.status = StepStatus::fPostStepDoItProc;
    rep.process = ProcessId::fHadronElastic;
    // THE PARTIAL SUMS THIS READS ARE THE ELASTIC DATA STORE'S, AT THIS ENERGY. In stage 1 they
    // were filled above by the process that won; in the final stage nothing has filled them yet,
    // and `G4NeutronGeneralProcess::PostStepDoIt` fills them here - `fCurrentXSS->
    // ComputeCrossSection`, and ONLY for a material with more than one element, because
    // `SampleZandA`'s element loop is skipped otherwise and reads nothing.
    if (sub_from_general && s.materials[mat].n_elements > 1) {
      (void)had::neutron_sub_xs_per_volume<real_t>(had.neutron.elastic, s.materials[mat],
                                                   p.ekin, loge, mxs_el);
    }
    const auto er = had::neutron_elastic_apply<real_t>(*had.neutron.elastic, s.materials[mat],
                                                       p.ekin, loge, p.dir, s.range_cut,
                                                       mxs_el, rng);
    if (er.interacted) {
      p.dir = er.dir;
      p.ekin = er.energy;
      // Both deposits with the same value, as note 5 of elastic/elastic_process.cuh records:
      // a sub-threshold recoil is proposed as LOCAL and as NON-IONIZING, which is what makes
      // elastic scattering contribute to NIEL and not to dose in a scorer that separates them.
      if (er.edep > real_t(0)) {
        if (scores) { edep += er.edep; }
        rep.non_ionizing += er.edep;
      }
      if (er.emit_recoil) {
        // The recoil of a neutron elastic scatter is the same population P8c's kernel
        // transports: hydrogen as a proton, the five light nuclei as themselves, everything
        // heavier as `kGenericIon` carrying its own (Z, A).
        em.push_nucleus(er.recoil_z, er.recoil_a, er.recoil_dir, er.recoil_ekin, p.event);
      }
      if (er.dropped_secondaries > 0) {
        had::book_refusal<real_t>(had.books, had::HadronicRefusal::kElasticDropped,
                                  er.recoil_ekin);
      }
    }
    if (!er.primary_survives) {
      // `efinal == 0`, which an elastic scatter off hydrogen at 180 degrees produces. Geant4
      // proposes fStopButAlive here (the neutron has `G4Decay` on its at-rest vector) and keeps
      // a zero-energy track; this transport cannot step one, because `TrackState::advance`
      // moves the clock by `L/(beta c)`. Killed with nothing left to deposit - which is where
      // Geant4's track ends up as well, one step later and through the 10 us cut.
      rep.status = StepStatus::fStopAndKill;
      p.ekin = real_t(0);
      return false;
    }
    return true;
  }

  // ---- capture. G4NeutronRadCapture, through P3's photon-evaporation cascade.
  //
  // The neutron always dies: `SetStatusChange(stopAndKill)` is the second line of
  // `ApplyYourself`, and it dies even on the branch that emits nothing (an unbound capture -
  // H3 + n has Q = -1.60 MeV, which is the case in P7's oracle grid where it fires).
  rep.status = StepStatus::fStopAndKill;
  rep.process = ProcessId::fNeutronCapture;
  if (sub_from_general && s.materials[mat].n_elements > 1) {
    (void)had::neutron_sub_xs_per_volume<real_t>(had.neutron.capture, s.materials[mat], p.ekin,
                                                 loge, mxs_cap);
  }
  {
    const auto cr = had::neutron_capture_apply<real_t>(
        *had.neutron.capture, s.materials[mat], p.ekin, loge, p.dir, p.global_time, p.weight,
        had.level_data, mxs_cap, rng, em, p.event);
    if (cr.edep > real_t(0) && scores) { edep += cr.edep; }
    if (cr.overflow > 0) {
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kCaptureOverflow, p.ekin);
    }
    if (cr.unmapped > 0) {
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kCaptureSecondarySpecies,
                                p.ekin);
    }
    // `CaptureRefusal::kUnphysicalTarget` is 3 and `kIsomerIonMass` is 1; only the first and
    // the resample limit mean the answer is missing something. See the enum's own note.
    if (cr.refused_resample_limit
        || cr.capture_refusal
               == static_cast<int>(
                      physics::hadronic::capture::CaptureRefusal::kUnphysicalTarget)) {
      had::book_refusal<real_t>(had.books, had::HadronicRefusal::kCaptureRefused, p.ekin);
    }
  }
  p.ekin = real_t(0);
  return false;
}

}  // namespace g4gpu
