// Heavy charged particle ionisation, transcribed from G4BetheBlochModel (11.1.1).
//
// G4MuIonisation, G4hIonisation and G4ionIonisation all use this model above a low-energy
// boundary (2 MeV scaled by the mass ratio; G4BraggModel / G4BraggIonModel below it), so
// this one formula covers muons, pions, kaons, protons, antiprotons and ions over most of
// their range.
//
// Not transcribed, and each is a real gap rather than a judgement call - see docs/RISK.md:
//   - G4EmCorrections::ShellCorrection      (matters below ~10 MeV/nucleon)
//   - G4EmCorrections::HighOrderCorrections (Barkas, Bloch and Mott terms)
//   - G4EmCorrections::IonBarkasCorrection  (the ion branch of the same)
//   - the ICRU90 tabulated stopping powers, used for protons and alphas below 2 MeV in
//     water, air and graphite only
// tests/test_hadron.cu measures what each omission costs against Geant4's own tables rather
// than asserting they are negligible.
#pragma once
#include <cmath>
#include "core/particle.cuh"
#include "core/units.cuh"
#include "data/materials.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/em_corrections.cuh"

namespace g4gpu::em {

/// The ionisation prefactor lives in electron_processes.cuh, which this header's users
/// already include through the physics list. It was defined here as well - the same quantity,
/// computed the same way, in the same namespace - until a test included both files.

/// Largest kinetic energy transferable to an atomic electron.
/// Verbatim from G4BetheBlochModel::MaxSecondaryEnergy.
template <typename real_t>
__host__ __device__ inline real_t hadron_max_secondary_energy(const ParticleDef<real_t>& pd,
                                                              real_t kinetic) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tau = kinetic / pd.mass;
  const real_t ratio = me / pd.mass;
  return real_t(2) * me * tau * (tau + real_t(2))
         / (real_t(1) + real_t(2) * (tau + real_t(1)) * ratio + ratio * ratio);
}

/// The nuclear form-factor cut on the delta-ray spectrum.
/// Verbatim from G4BetheBlochModel::SetupParameters (the LeptonNumber == 0 branch, i.e.
/// hadrons and ions; leptons keep tlimit = infinity).
template <typename real_t>
__host__ __device__ inline real_t hadron_tlimit(const ParticleDef<real_t>& pd,
                                                ParticleType type) {
  const bool is_lepton = (type == ParticleType::kMuonMinus || type == ParticleType::kMuonPlus);
  if (is_lepton) { return real_t(1e30); }
  const real_t me = units::electron_mass_c2<real_t>();
  real_t x = real_t(842.6);  // 0.8426 GeV, in MeV
  if (pd.spin == real_t(0) && pd.mass < real_t(1000)) {
    x = real_t(736.0);
  } else if (pd.mass > real_t(1000)) {
    const int iz = static_cast<int>(fabs(pd.charge) + real_t(0.5));
    if (iz > 1) { x /= pow(real_t(iz), real_t(0.27)); }  // G4NistManager::GetA27
  }
  const real_t formfact = real_t(2) * me / (x * x);
  return real_t(2) / formfact;
}

/// Restricted dE/dx, MeV/mm. Verbatim from G4BetheBlochModel::ComputeDEDXPerVolume, minus
/// the shell and high-order corrections and the ICRU90 low-energy branch (see the file
/// comment).
template <typename real_t>
__host__ __device__ inline real_t bethe_bloch_dedx(const data::Material<real_t>& m,
                                                   ParticleType type, real_t kinetic,
                                                   real_t cut,
                                                   const ShellTables<real_t>* shell = nullptr) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t tlim = hadron_tlimit(pd, type);
  const real_t cut_energy = fmin(fmin(cut, tmax), tlim);

  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t xc = cut_energy / tmax;
  const real_t eexc = m.mean_excitation;  // MeV
  const real_t eexc2 = eexc * eexc;
  const real_t charge_square = pd.charge * pd.charge;

  real_t dedx = log(real_t(2) * me * bg2 * cut_energy / eexc2) - (real_t(1) + xc) * beta2;
  if (pd.spin > real_t(0)) {
    const real_t del = real_t(0.5) * cut_energy / (kinetic + pd.mass);
    dedx += del * del;
  }
  // Sternheimer density effect, the same G4IonisParamMat::DensityCorrection the
  // electron ionisation already uses.
  const real_t twoln10 = real_t(2) * log(real_t(10));
  const real_t x = log(bg2) / twoln10;
  dedx -= data::density_correction(m, x);
  // The shell correction enters twice, as G4BetheBlochModel writes it.
  if (shell != nullptr) {
    dedx -= real_t(2) * shell_correction(*shell, m, pd, kinetic);
  }
  dedx *= twopi_mc2_rcl2<real_t>() * charge_square * m.electron_density / beta2;
  // Geant4 applies the same prefactor to the high-order terms, so they are added after the
  // multiply rather than inside the bracket. Ions take only 2*Barkas (IonBarkasCorrection);
  // everything else takes 2*(Barkas + Bloch) + Mott (HighOrderCorrections).
  //
  // Geant4 substitutes an effective charge for the correction terms when |q| > 1.5
  // (G4EmCorrections::SetupKinematics -> G4ionEffectiveCharge); that model is not
  // transcribed, so alpha and He3 use the nominal charge here. See docs/RISK.md.
  if (shell != nullptr) {
    // G4EmCorrections::SetupKinematics substitutes the effective charge whenever |q| > 1.5,
    // and its q2 - not the nominal charge squared - scales the correction prefactor.
    ParticleDef<real_t> cpd = pd;
    if (fabs(pd.charge) > real_t(1.5)) {
      cpd.charge = ion_effective_charge(m, pd, kinetic);
    }
    const real_t q2c = cpd.charge * cpd.charge;
    const real_t pref = twopi_mc2_rcl2<real_t>() * q2c * m.electron_density / beta2;
    dedx += (pd.is_ion ? real_t(2) * barkas_correction(m, cpd, kinetic)
                       : high_order_bracket(m, cpd, kinetic))
            * pref;
  }
  return fmax(dedx, real_t(0));
}

/// Delta-ray production cross section per volume, 1/mm.
/// Transcribed from G4BetheBlochModel::ComputeCrossSectionPerElectron scaled by the
/// electron density, which is what CrossSectionPerVolume does.
template <typename real_t>
__host__ __device__ inline real_t bethe_bloch_delta_xs(const data::Material<real_t>& m,
                                                       ParticleType type, real_t kinetic,
                                                       real_t cut, real_t max_energy) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t maxE = fmin(max_energy, tmax);
  const real_t cutE = fmin(cut, maxE);
  if (cutE >= maxE) { return real_t(0); }

  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t charge_square = pd.charge * pd.charge;

  real_t cross = real_t(1) / cutE - real_t(1) / maxE
                 - beta2 * log(maxE / cutE) / tmax;
  if (pd.spin > real_t(0)) {
    const real_t energy = kinetic + pd.mass;
    cross += real_t(0.5) * (maxE - cutE) / (energy * energy);
  }
  cross *= units::twopi<real_t>() * units::classic_electron_radius<real_t>()
           * units::classic_electron_radius<real_t>() * me * charge_square / beta2;
  return fmax(cross, real_t(0)) * m.electron_density;
}

/// The boundary G4hIonisation / G4ionIonisation put between the Bragg models and
/// Bethe-Bloch: 2 MeV for a proton, scaled by the mass ratio for anything heavier.
template <typename real_t>
__host__ __device__ inline real_t bragg_bethe_boundary(const ParticleDef<real_t>& pd) {
  constexpr real_t kProtonMass = units::proton_mass_c2<real_t>();
  constexpr real_t kBoundaryForProton = real_t(2.0);  // MeV
  return kBoundaryForProton * pd.mass / kProtonMass;
}


/// 8-point Gauss-Legendre nodes G4MuBetheBlochModel integrates the radiative correction on.
template <typename real_t> __host__ __device__ inline const real_t* mu_xgi() {
  static const real_t v[8] = {real_t(0.0199), real_t(0.1017), real_t(0.2372), real_t(0.4083),
                              real_t(0.5917), real_t(0.7628), real_t(0.8983), real_t(0.9801)};
  return v;
}
template <typename real_t> __host__ __device__ inline const real_t* mu_wgi() {
  static const real_t v[8] = {real_t(0.0506), real_t(0.1112), real_t(0.1569), real_t(0.1813),
                              real_t(0.1813), real_t(0.1569), real_t(0.1112), real_t(0.0506)};
  return v;
}
/// Energy above which the radiative correction to muon ionisation is integrated.
template <typename real_t> __host__ __device__ constexpr real_t kMuLimitKinEnergy() {
  return real_t(0.1);  // 100 keV
}
/// alpha / 2 pi.
template <typename real_t> __host__ __device__ inline real_t alpha_prime() {
  return units::fine_structure_const<real_t>() / units::twopi<real_t>();
}

/// Muon ionisation dE/dx, transcribed from G4MuBetheBlochModel::ComputeDEDXPerVolume.
///
/// G4MuIonisation uses this above 200 keV rather than G4BetheBlochModel. It differs by a
/// radiative correction: an O(alpha) term for delta rays above 100 keV, in which the
/// knock-on electron radiates. The muon is light enough for that to matter.
template <typename real_t>
__host__ __device__ inline real_t mu_bethe_bloch_dedx(const data::Material<real_t>& m,
                                                      ParticleType type, real_t kinetic,
                                                      real_t cut,
                                                      const ShellTables<real_t>* shell =
                                                          nullptr) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t cut_energy = fmin(cut, tmax);
  const real_t tau = kinetic / pd.mass;
  const real_t gam = tau + real_t(1);
  const real_t bg2 = tau * (tau + real_t(2));
  const real_t beta2 = bg2 / (gam * gam);
  const real_t eexc2 = m.mean_excitation * m.mean_excitation;

  real_t dedx = log(real_t(2) * me * bg2 * cut_energy / eexc2)
                - (real_t(1) + cut_energy / tmax) * beta2;
  const real_t tot = kinetic + pd.mass;
  const real_t del = real_t(0.5) * cut_energy / tot;
  dedx += del * del;
  const real_t twoln10 = real_t(2) * log(real_t(10));
  dedx -= data::density_correction(m, log(bg2) / twoln10);
  if (shell != nullptr) { dedx -= real_t(2) * shell_correction(*shell, m, pd, kinetic); }
  dedx = fmax(dedx, real_t(0));

  // O(alpha) radiative correction to the knock-on spectrum.
  if (cut_energy > kMuLimitKinEnergy<real_t>()) {
    const real_t logtmax = log(cut_energy);
    const real_t logstep = logtmax - log(kMuLimitKinEnergy<real_t>());
    const real_t ftot2 = real_t(0.5) / (tot * tot);
    const real_t mass2 = pd.mass * pd.mass;
    const real_t* xgi = mu_xgi<real_t>();
    const real_t* wgi = mu_wgi<real_t>();
    real_t dloss = real_t(0);
    for (int l = 0; l < 8; ++l) {
      const real_t ep = exp(log(kMuLimitKinEnergy<real_t>()) + xgi[l] * logstep);
      const real_t a1 = log(real_t(1) + real_t(2) * ep / me);
      const real_t a3 = log(real_t(4) * tot * (tot - ep) / mass2);
      dloss += wgi[l] * (real_t(1) - beta2 * ep / tmax + ep * ep * ftot2) * a1 * (a3 - a1);
    }
    dedx += dloss * logstep * alpha_prime<real_t>();
  }
  dedx *= twopi_mc2_rcl2<real_t>() * m.electron_density / beta2;
  if (shell != nullptr) {
    dedx += high_order_bracket(m, pd, kinetic) * twopi_mc2_rcl2<real_t>()
            * pd.charge * pd.charge * m.electron_density / beta2;
  }
  return fmax(dedx, real_t(0));
}

/// Muon delta-ray cross section per volume, 1/mm.
/// Transcribed from G4MuBetheBlochModel::ComputeCrossSectionPerElectron, scaled by the
/// electron density.
template <typename real_t>
__host__ __device__ inline real_t mu_bethe_bloch_delta_xs(const data::Material<real_t>& m,
                                                          ParticleType type, real_t kinetic,
                                                          real_t cut, real_t max_energy) {
  const ParticleDef<real_t> pd = particle_def<real_t>(type);
  if (pd.mass <= real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tmax = hadron_max_secondary_energy(pd, kinetic);
  const real_t maxE = fmin(tmax, max_energy);
  if (cut >= maxE) { return real_t(0); }
  const real_t tot = kinetic + pd.mass;
  const real_t e2 = tot * tot;
  const real_t beta2 = kinetic * (kinetic + real_t(2) * pd.mass) / e2;

  real_t cross = real_t(1) / cut - real_t(1) / maxE - beta2 * log(maxE / cut) / tmax
                 + real_t(0.5) * (maxE - cut) / e2;
  if (maxE > kMuLimitKinEnergy<real_t>()) {
    const real_t logtmax = log(maxE);
    const real_t logtmin = log(fmax(cut, kMuLimitKinEnergy<real_t>()));
    const real_t logstep = logtmax - logtmin;
    const real_t mass2 = pd.mass * pd.mass;
    const real_t* xgi = mu_xgi<real_t>();
    const real_t* wgi = mu_wgi<real_t>();
    real_t dcross = real_t(0);
    for (int l = 0; l < 8; ++l) {
      const real_t ep = exp(logtmin + xgi[l] * logstep);
      const real_t a1 = log(real_t(1) + real_t(2) * ep / me);
      const real_t a3 = log(real_t(4) * tot * (tot - ep) / mass2);
      dcross += wgi[l] * (real_t(1) / ep - beta2 / tmax + real_t(0.5) * ep / e2) * a1
                * (a3 - a1);
    }
    cross += dcross * logstep * alpha_prime<real_t>();
  }
  cross *= twopi_mc2_rcl2<real_t>() / beta2;
  return fmax(cross, real_t(0)) * m.electron_density;
}

/// The energy at which G4MuIonisation hands over from the Bragg/ICRU73QO models to
/// G4MuBetheBlochModel (G4MuIonisation::InitialiseEnergyLossProcess uses 200 keV).
template <typename real_t> __host__ __device__ constexpr real_t kMuBetheBlochLow() {
  return real_t(0.2);  // MeV
}

}  // namespace g4gpu::em
