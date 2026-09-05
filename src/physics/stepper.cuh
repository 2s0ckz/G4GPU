// Single-step physics for the track-parallel scheduler.
//
// Restructured from "follow one track to
// completion" into "advance one track by one step". That is what removes the per-thread
// secondary stack and the loop-carried state that drove register pressure to 108 in the
// thread-per-event kernel.
#pragma once
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
#include "physics/scene.cuh"
#include "render/trajectory.cuh"

namespace g4gpu {

/// Advances one photon by a single step.
/// @param edep energy deposited in the scoring volume by this step, MeV
/// @return true if the photon is still alive and should be requeued
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_gamma(const Scene<real_t>& s, TrackState<real_t>& p, Rng& rng,
                                  Emitter& em, real_t& edep,
                                  vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld) { return false; }

  if (p.ekin < em::kPhotonAbsorbCut<real_t>()) {
    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = p.ekin; }
    return false;
  }

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
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
    p.pos = p.pos + (d_boundary + geom::kPushDistance<real_t>()) * p.dir;
    traj.add(pos_before, p.pos, ParticleType::kGamma, p.event);
    p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
    return p.volume != geom::kOutsideWorld;
  }

  p.pos = p.pos + s_int * p.dir;
  traj.add(pos_before, p.pos, ParticleType::kGamma, p.event);
  em.pos = p.pos;
  em.volume = p.volume;
  em.event = p.event;

  const auto proc = em::select_gamma_process(xs, rng.uniform());

  if (proc == em::GammaProcess::kCompton) {
    const auto r = em::sample_klein_nishina<real_t>(
        p.ekin, p.dir, rng, em, p.event, em::kElectronTrackingCut<real_t>(), real_t(1e-6));
    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = r.local_deposit; }
    if (!r.gamma_survives) { return false; }
    p.ekin = r.gamma_energy;
    p.dir = r.gamma_dir;
    return true;
  }

  if (proc == em::GammaProcess::kRayleigh && s.rayleigh != nullptr) {
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
                                   Rng& rng, Emitter& em, real_t& edep,
                                   vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
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

    // A discrete process fires only if it was the limiting true length and geometry did not
    // cut in first.
    const bool annihilates =
        !hits_boundary && (d_annih <= t_step) && (d_annih <= d_brem) && (d_annih <= d_delta);
    const bool emits_brem =
        !hits_boundary && !annihilates && (d_brem <= t_step) && (d_brem <= d_delta);
    const bool emits_delta = !hits_boundary && !annihilates && !emits_brem && (d_delta <= t_step);

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
    traj.add(pos_before, p.pos,
             is_positron ? ParticleType::kPositron : ParticleType::kElectron, p.event);

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
///   * **The loss is all local.** A hadron has no radiative split worth making at these
///     energies. G4hBremsstrahlung and G4hPairProduction are registered by the physics list,
///     but the oracle's dedx_brem and dedx_pair columns are zero for every proton and alpha row
///     below a GeV, so there is no share to split off.
///
///   * **There are no nuclear interactions.** This is EM transport: a proton here is stopped by
///     electrons, never by a nucleus. That is a real omission with a known size rather than an
///     approximation - see docs/RISK.md.
template <typename real_t, typename Rng, typename Emitter>
__device__ inline bool step_hadron(const Scene<real_t>& s, TrackState<real_t>& p,
                                   ParticleType type, Rng& rng, Emitter& em, real_t& edep,
                                   vis::TrajectoryBuffer traj = vis::no_capture()) {
  edep = real_t(0);
  const Vec3<real_t> pos_before = p.pos;
  if (p.volume == geom::kOutsideWorld || s.hadron_range == nullptr) { return false; }

  const em::HadronSpecies sp = em::hadron_species_of<real_t>(type);
  const ParticleDef<real_t> pd = particle_def<real_t>(type);

  const int mat = geom::material_at(s.geometry, p.volume, p.pos);
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

    // A discrete process fires only if it was the limiting true length and geometry did not
    // cut in first.
    const bool emits_delta = !hits_boundary && (d_delta <= t_step);

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
    if (e_after > real_t(0) && step_len > real_t(0)) {
      const real_t t_mean = real_t(0.5) * (e_before + e_after);
      const real_t nl = fmin(step_len * em::nuclear_stopping_dedx(mm, type, t_mean), e_before);
      if (nl > real_t(0)) {
        e_after = fmax(e_after - nl, real_t(0));
        loss = e_before - e_after;
      }
    }

    if (s.geometry.volumes[p.volume].score_index >= 0) { edep = loss; }
    p.ekin = e_after;

    p.pos = p.pos + geom_step * p.dir;
    if (hits_boundary) {
      p.pos = p.pos + geom::kPushDistance<real_t>() * p.dir;
      p.volume = geom::resolve_after_step(s.geometry, next_volume, p.pos);
    }
    traj.add(pos_before, p.pos, type, p.event);

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
  if (p.volume >= 0 && s.geometry.volumes[p.volume].score_index >= 0) { edep += p.ekin; }
  return false;
}

}  // namespace g4gpu
