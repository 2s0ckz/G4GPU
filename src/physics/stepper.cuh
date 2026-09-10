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
#include "physics/em/wentzel_msc.cuh"
#include "physics/em/gamma_processes.cuh"
#include "physics/em/klein_nishina.cuh"
#include "physics/em/annihilation.cuh"
#include "physics/em/pair_production.cuh"
#include "physics/em/urban_msc.cuh"
#include "physics/hadronic/neutron_general_xs.cuh"
#include "physics/scene.cuh"
#include "render/trajectory.cuh"

namespace g4gpu {

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
    t_step = fmin(t_step, range);

    // lambda at the energy left after the whole true step, for the general geometric branch.
    real_t lambda1 = real_t(-1);
    {
      const real_t rfin = fmax(range - t_step, real_t(0.01) * range);
      const real_t e_rfin = s.range_table->energy_from_range(mat, rfin);
      if (e_rfin > real_t(0)) { lambda1 = s.msc->lambda_at(mat, is_positron, e_rfin); }
    }
    em::MscStep<real_t> msc_state;
    const real_t z_step = em::urban_geom_path(t_step, lambda0, range, lambda1, p.ekin, msc_state);

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
///   * **Multiple scattering is WentzelVI, not Urban.** Different model, different step limit,
///     and `facrange` is 0.2 rather than 0.04, which is a factor of five on the step length.
///     Its lateral displacement is switched off for a heavy particle
///     (G4EmParameters::MuHadLateralDisplacement is false), so a proton is deflected but never
///     shifted sideways - the branch step_lepton spends twenty lines on does not exist here.
///
///     **This is right for the proton and wrong for the alpha.** Geant4 prints its own answer
///     at initialisation, and it reads:
///
///         msc:  for proton  SubType= 10
///                 WentzelVIUni : Emin=    0 eV  Emax=  100 TeV
///         msc:  for alpha  SubType= 10
///                     UrbanMsc : Emin=    0 eV  Emax=  100 TeV
///
///     G4EmBuilder::ConstructCharged gives an ion `G4hMultipleScattering("ionmsc")` carrying
///     G4UrbanMscModel, and only muons and singly-charged hadrons get WentzelVI. Using
///     WentzelVI for the alpha here is a substitution, not a transcription.
///
///     **Which species take the substitution is now a predicate**, `uses_wentzel_msc` in
///     core/particle.cuh, because wiring nine more species made it a list rather than a
///     footnote about the alpha. WentzelVI is correct for mu+-, pi+-, K+-, p and pbar; it is a
///     substitution for alpha, He3, GenericIon and - newly - the deuteron and the triton,
///     which share the physics list's one model-less `G4hMultipleScattering("ionmsc")`.
///
///     It is not fixed because urban_msc.cuh is transcribed for leptons specifically - its
///     step limit and its sampler both take `is_positron`, and its transport mean free path
///     comes from an e-/e+ table - so an alpha would need the model generalised to arbitrary
///     mass and charge, which is the model itself rather than a dispatch. What the
///     substitution costs is measured rather than assumed: `tools\compare_b1_beams.ps1` runs
///     example B1 with an 840 MeV alpha against a real Geant4 build of the same example and
///     the doses agree to about a per cent, which bounds it. MSC moves a track sideways; the
///     quantity there is the energy deposited in a 12 cm wide volume, and a few milliradians
///     of difference in how a track wanders does not move it.
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
///     as long as it has been stepped. docs/PORTED.md 1.3 and docs/RISK.md V36 carry the numbers
///     and the reason no energy refusal was added.
///
///   * **There are no nuclear interactions.** This is EM transport: a proton here is stopped by
///     electrons, never by a nucleus. That is a real omission with a known size rather than an
///     approximation - see docs/RISK.md.
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_hadron(const Scene<real_t>& s, TrackState<real_t>& p,
                                   ParticleType type, Rng& rng, Emitter& em, real_t& edep,
                                   StepReport<real_t>& rep,
                                   vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  rep = StepReport<real_t>{};
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld || s.hadron_range == nullptr) { return false; }

  const em::HadronSpecies sp = em::hadron_species_of<real_t>(type);
  const ParticleDef<real_t> pd = particle_def<real_t>(type);

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
  rep.material = mat;
  const real_t range = s.hadron_range->lookup(sp, mat, p.ekin);

  // The same two termination guards step_lepton needs, for the same reason: without them a
  // track can stop making progress near the end of its range and never fall below the cut, and
  // the drain loop never empties. 10 um of residual range is a few tens of keV for a proton.
  const real_t kMinUsefulRange = real_t(1e-2);  // mm
  if (p.ekin >= em::kHadronTrackingCut<real_t>() && range >= kMinUsefulRange) {
    const data::Material<real_t>& mm = s.materials[mat];
    const real_t cut = mm.cut_electron;

    int next_volume = geom::kOutsideWorld;
    const real_t d_boundary =
        geom::step_to_boundary(s.geometry, p.volume, p.pos, p.dir, next_volume);

    // Discrete delta-ray production competes with the boundary and the step limits.
    const real_t delta_xs = em::hadron_delta_xs(mm, type, p.ekin, cut, real_t(1e30));
    const real_t d_delta = (delta_xs > real_t(0)) ? -log(rng.uniform()) / delta_xs
                                                  : geom::kInfinity<real_t>();

    // Continuous-loss limit, G4VEnergyLossProcess::AlongStepGetPhysicalInteractionLength with
    // the mu/hadron step function (0.2, 0.1 mm).
    const real_t finR = em::kHadronFinalRange<real_t>();
    const real_t dRoR = em::kHadronDRoverRange<real_t>();
    const real_t max_step =
        (range > finR) ? range * dRoR + finR * (real_t(1) - dRoR) * (real_t(2) - finR / range)
                       : range;

    // ---- WentzelVI multiple scattering.
    //
    // No table: the transport cross section is closed-form, so lambda comes straight from
    // wv_transport_xs. That call also fills the per-element single-scattering tables the
    // sampler needs to pick which atom a discrete scatter happened on, which is why it is made
    // even where only lambda is wanted.
    constexpr real_t kCosThetaLim = real_t(-1);  // G4EmParameters::MscThetaLimit() = pi
    em::WentzelMscState<real_t> st{};
    em::WentzelElementXs<real_t> els{};
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
    em::wv_transport_xs(mm, pd, type, p.ekin, cut, kCosThetaLim, real_t(1), st.cos_tet_max_nuc,
                        els, st.xtsec);
    st.lambda_eff = em::wentzel_lambda(mm, type, p.ekin, cut, kCosThetaLim);

    const real_t safety = geom::compute_safety(s.geometry, p.volume, p.pos);
    rep.safety = safety;
    const real_t t_msc =
        s.processes.multiple_scattering
            ? em::wv_step_limit(mm, pd, type, p.ekin, range, st.lambda_eff, st.cos_tet_max_nuc,
                                kCosThetaLim, safety, s.range_cut, max_step,
                                em::kHadronFacRange<real_t>())
            : geom::kInfinity<real_t>();

    // The true path length this step would take if geometry did not interrupt it.
    real_t t_step = fmin(fmin(max_step, t_msc), d_delta);
    t_step = fmin(t_step, range);

    // The energy after the whole true step, and the transport mfp at the mean energy - both
    // are inputs to the true/geometric conversion, so both are computed from the uninterrupted
    // step before geometry can cut it short.
    const real_t e_end =
        s.hadron_range->energy_from_range(sp, mat, fmax(range - t_step, real_t(0)));
    real_t lambda_eff_end = st.lambda_eff;
    real_t cos_tet_max_end = st.cos_tet_max_nuc;
    {
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
        lambda_eff_end = em::wentzel_lambda(mm, type, e_mid, cut, kCosThetaLim);
      }
    }

    const real_t z_step =
        s.processes.multiple_scattering
            ? em::wv_geom_path(st, t_step, e_end, lambda_eff_end, cos_tet_max_end)
            : t_step;

    const bool hits_boundary = (d_boundary < z_step);
    const real_t geom_step = hits_boundary ? d_boundary : z_step;

    real_t step_len = geom_step;
    if (s.processes.multiple_scattering) {
      auto recompute = [&](real_t cos_min, real_t& xt) {
        return em::wv_transport_xs(mm, pd, type, st.eff_kin_energy, cut, kCosThetaLim, cos_min,
                                   st.cos_tet_max_nuc, els, xt);
      };
      step_len =
          em::wv_true_path(st, geom_step, e_end, lambda_eff_end, cos_tet_max_end, recompute);
    }
    // See the note in step_lepton: the true path, which is what the energy loss below is
    // computed against and what G4Step::GetStepLength reports.
    rep.true_length = step_len;

    // A discrete process fires only if it was the limiting true length and geometry did not
    // cut in first.
    const bool emits_delta = !hits_boundary && (d_delta <= t_step);

    // Read back out as G4StepPoint::GetProcessDefinedStep would report it. See the same block
    // in step_lepton.
    if (hits_boundary) {
      rep.status = StepStatus::fGeomBoundary;
      rep.process = ProcessId::fTransportation;
    } else if (emits_delta) {
      rep.status = StepStatus::fPostStepDoItProc;
      rep.process = ProcessId::fIonisation;
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
    constexpr real_t kLinLossLimit = real_t(0.01);  // G4EmParameters::LinearLossLimit
    if (step_len >= range || e_before <= em::kHadronTrackingCut<real_t>()) {
      loss = e_before;
    } else {
      loss = step_len * s.hadron_range->dedx_at(sp, mat, e_before);
      if (loss > e_before * kLinLossLimit) {
        loss = e_before
               - s.hadron_range->energy_from_range(sp, mat, fmax(range - step_len, real_t(0)));
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
          // The effective charge squared is the bare charge squared for an alpha:
          // G4IonFluctuations only ever sees an effective charge through
          // SetParticleAndCharge, which G4VEnergyLossProcess calls under `if(isIon)`, and
          // G4EmTableUtil::CheckIon leaves isIon false for deuteron, triton, alpha+ and alpha
          // by name. A generic ion will need the real ratio here; it is a parameter rather
          // than a constant inside the model for exactly that reason.
          loss = em::sample_ion_fluctuation(mm, pd, e_before, tcut, tmax, step_len, loss,
                                            pd.charge * pd.charge, rng);
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
    }
    traj.add(pos_before, p.pos, type, p.event, p.rng_key);

    // ---- the scattering itself. lat_displacement is false for every heavy particle, so the
    // returned displacement is zero and there is nothing to apply.
    if (s.processes.multiple_scattering) {
      const auto sc = em::wv_sample_scattering(mm, pd, type, st, els, cut, kCosThetaLim, p.dir,
                                               em::kHadronLateralDisplacement<real_t>(), rng);
      p.dir = sc.dir;
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

    if (p.ekin >= em::kHadronTrackingCut<real_t>() && p.volume != geom::kOutsideWorld) {
      return true;
    }
  }

  // Dying: whatever is left is deposited here. A stopped proton in this transport does not
  // capture on a nucleus - there is no hadronic physics - so there is nothing else to do.
  rep.status = StepStatus::fStopAndKill;
  if (rep.process == ProcessId::fNotDefined) { rep.process = ProcessId::fBelowTrackingCut; }
  if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += p.ekin; }
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
/// @param xs the combined table, or null. Null is the state before P8: the cross section is
///        then zero, `s_int` is infinite, and the neutron streams to the world boundary - which
///        is what a Geant4 neutron does with `NeutronGeneralProc` inactivated, and the only
///        honest thing to do with a process that does not exist yet.
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_neutral(const Scene<real_t>& s, TrackState<real_t>& p,
                                    ParticleType type,
                                    const had::NeutronGeneralXs<real_t>* xs, Rng& rng,
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

  // Zero for a pi0 and for a neutron with no table, which is every run today.
  const real_t sigma = (xs != nullptr && type == ParticleType::kNeutron)
                           ? xs->total(mat, p.ekin) : real_t(0);
  const real_t s_int =
      (sigma > real_t(0)) ? -log(rng.uniform()) / sigma : geom::kInfinity<real_t>();

  if (s_int >= d_boundary) {
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

  // ---- a sub-process fired.
  //
  // UNREACHABLE TODAY, and it says so in the only way the build can hear. `sigma` is zero
  // unless a table is present, and TransportEngine::Upload refuses to hold a table without the
  // final states to go with it - so nothing can arrive here until P8 has written both. When it
  // has, this is where `xs->select(mat, p.ekin, rng.uniform())` chooses between elastic,
  // inelastic and capture and the chosen model's final state is applied.
  //
  // Until then the step is left annotated with `fNotDefined`, which is not laziness: it is the
  // one value `g4dose -verify-step-hook` fails the build on, and it is reserved for exactly
  // this - a branch nobody has annotated because nobody has written it. A track reaching here
  // is killed with its energy deposited locally, which is the conservative disposal, but the
  // verifier refuses the run before that number can be used for anything.
  rep.true_length = s_int;
  rep.status = StepStatus::fStopAndKill;
  rep.process = ProcessId::fNotDefined;
  p.pos = p.pos + s_int * p.dir;
  traj.add(pos_before, p.pos, type, p.event, p.rng_key);
  if (s.geometry.volumes[p.volume].score_index >= 0) { edep = p.ekin; }
  return false;
}

}  // namespace g4gpu
