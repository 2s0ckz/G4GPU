// Photon evaporation: the giant-resonance continuum probability, the discrete cascade over
// tabulated levels, internal conversion, and the two-body transition kinematics.
//
// Transcribed from G4PhotonEvaporation and G4GammaTransition (11.1.1). This is channel 0 of
// the 68 the default evaporation factory offers, and it is the one that finishes almost every
// de-excitation: below the lowest particle-emission threshold it is the only channel with a
// non-zero probability, and G4Evaporation then hands the fragment to BreakUpChain to walk it
// to the ground state in one go.
//
// What the default flags remove here is worth stating, because it is most of the class:
//
//   fCorrelatedGamma = false  no G4NuclearPolarization, so G4GammaTransition::SampleDirection
//                             is G4RandomDirection() and G4PolarizationTransition - 403 lines
//                             of Clebsch-Gordan algebra - is never constructed. Refused by
//                             name below.
//   fStoreAllLevels = false   G4LevelReader builds no shell-probability table, so
//                             G4NucLevel::SampleShell always returns -1, so
//                             G4GammaTransition::SampleTransition takes bond_energy = 0 and
//                             never reads G4AtomicShells. Internal conversion still HAPPENS -
//                             fICM is true - it just emits the electron with the full
//                             transition energy. That is not an approximation of this port's:
//                             it is what the configuration computes.
//   fRDM = false              this is not radioactive decay, so the level lifetime is sampled
//                             (fSampleTime) and the nuclear-polarization store is untouched.
//
// The continuum probability is a Lorentzian giant dipole resonance times a level-density
// factor, integrated on a grid of at most MAXDEPOINT = 10 points and at most 1 MeV apart. Ten
// points is few, and the cumulative array it leaves is what the gamma energy is later sampled
// from by linear interpolation - so the coarseness is part of the spectrum, not an error in it.
#ifndef G4GPU_DEEX_PHOTON_EVAPORATION_CUH
#define G4GPU_DEEX_PHOTON_EVAPORATION_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/g4pow.hh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

constexpr int kMaxDePoint = 10;   ///< G4PhotonEvaporation MAXDEPOINT
constexpr int kMaxGRData = 300;   ///< G4PhotonEvaporation MAXGRDATA

/// PDG codes for the two things photon evaporation can emit. The handler passes these out as
/// codes rather than as a ParticleType, because core/particle.cuh belongs to another package.
constexpr int kPdgGamma = 22;
constexpr int kPdgElectron = 11;

/// G4PhotonEvaporation::InitialiseGRData - the giant-resonance centroid and width per mass
/// number. `GREnergy[A] = 40.3 MeV / A^0.2` through G4Pow::powZ, `GRWidth = 0.3 GREnergy`.
///
/// Stored as FLOAT, because Geant4 stores them as G4float and then compares
/// `fExcEnergy >= GREfactor*GRWidth[A] + GREnergy[A]` against a double. Keeping them double
/// here would move that threshold by a part in 1e7 and change which fragments get a gamma
/// probability at all near the cut.
__host__ __device__ inline float gr_energy(int A) {
  int a = A;
  if (a < 1) { a = 1; }
  if (a >= kMaxGRData) { a = kMaxGRData - 1; }
  return static_cast<float>(40.3 * u::MeV<double>() / g4pow_pow_z(a, 0.2));
}
__host__ __device__ inline float gr_width(int A) {
  const float GRWfactor = 0.3f;
  return GRWfactor * gr_energy(A);
}

/// The state G4PhotonEvaporation keeps between GetEmissionProbability and GenerateGamma. The
/// coupling is real and is Geant4's: GenerateGamma recomputes the probability only when
/// `fCode != 1000*Z + A || eexc != fExcEnergy`, so a caller that changes the fragment between
/// the two calls gets the previous fragment's cumulative array.
struct PhotonEvaporationState {
  int code = 0;             ///< 1000*Z + A of the fragment the arrays belong to
  double exc_energy = 0.0;
  double probability = 0.0;
  double step = 0.0;
  int points = 0;
  double cumm[kMaxDePoint] = {0.0};
  int level_index = 0;      ///< G4PhotonEvaporation::fIndex, the level the cascade is on
  int man = -1;             ///< the level manager, or -1
  int man_z = 0, man_a = 0;
  double level_energy_max = 0.0;
};

/// G4PhotonEvaporation::InitialiseLevelManager - caches the manager for (Z, A) and resets the
/// level index when the nuclide changes.
__host__ __device__ inline void initialise_level_manager(PhotonEvaporationState& s,
                                                          const data::LevelTable& lt, int Z,
                                                          int A) {
  if (Z != s.man_z || A != s.man_a) {
    s.man_z = Z;
    s.man_a = A;
    s.level_index = 0;
    s.man = data::find_manager(lt, Z, A);
    s.level_energy_max = (s.man >= 0) ? data::read_max_level_energy(lt, s.man) : 0.0;
  }
}

/// G4PhotonEvaporation::GetEmissionProbability.
///
/// Four refusals of its own, all returning zero probability: A <= 1, Z <= 0, Z == A (a nucleus
/// of only protons has no gamma transitions in this dataset), and an excitation below the
/// 10 eV tolerance. Then one more: above `5 * GRWidth + GREnergy` the fragment is treated as
/// too highly excited for a gamma to compete, and the probability is zero - which is how a
/// 200 MeV excitation ends up emitting nucleons rather than photons.
///
/// `emax` is the neutron separation energy, and the comment in Geant4 says why: continuum
/// transitions are limited to final states below the Fermi energy, "this approach needs
/// further evaluation". It is the number that runs.
__host__ __device__ inline double photon_emission_probability(PhotonEvaporationState& s,
                                                               const Fragment& frag,
                                                               const data::LevelTable& lt) {
  s.probability = 0.0;
  s.exc_energy = frag.excitation;
  const int Z = frag.z;
  int A = frag.a;
  s.code = 1000 * Z + A;
  const double tolerance = deex_params().min_excitation;
  if (Z <= 0 || A <= 1 || Z == A || tolerance >= s.exc_energy) { return s.probability; }

  if (A >= kMaxGRData) { A = kMaxGRData - 1; }
  const float GREfactor = 5.0f;
  if (s.exc_energy >= static_cast<double>(GREfactor * gr_width(A) + gr_energy(A))) {
    return s.probability;
  }

  double emax = deex::nuclear_mass(frag.a - 1, Z) + u::neutron_mass_c2<double>() -
                frag.ground_state_mass;
  if (emax < 0.0) { emax = 0.0; }
  if (emax > s.exc_energy) { emax = s.exc_energy; }
  const double eexcfac = 0.99;
  if (0.0 == emax || s.exc_energy * eexcfac <= emax) { emax = s.exc_energy * eexcfac; }

  s.step = emax;
  const double MaxDeltaEnergy = u::MeV<double>();
  s.points = static_cast<int>(s.step / MaxDeltaEnergy) + 2;
  if (s.points > kMaxDePoint) { s.points = kMaxDePoint; }
  s.step /= static_cast<double>(s.points - 1);

  const double eres = static_cast<double>(gr_energy(A));
  const double wres = static_cast<double>(gr_width(A));
  const double eres2 = eres * eres;
  const double wres2 = wres * wres;
  const bool has_levels = (data::find_manager(lt, Z, frag.a) >= 0);
  const double level_dens = deex::level_density(Z, frag.a, s.exc_energy, has_levels);
  const double xsqr = std::sqrt(level_dens * s.exc_energy);

  double egam = s.exc_energy;
  double gammaE2 = egam * egam;
  double gammaR2 = gammaE2 * wres2;
  double egdp2 = gammaE2 - eres2;

  double p0 = std::exp(-2.0 * xsqr) * gammaR2 * gammaE2 / (egdp2 * egdp2 + gammaR2);
  double p1 = 0.0;
  for (int i = 0; i < kMaxDePoint; ++i) { s.cumm[i] = 0.0; }

  for (int i = 1; i < s.points; ++i) {
    egam -= s.step;
    gammaE2 = egam * egam;
    gammaR2 = gammaE2 * wres2;
    egdp2 = gammaE2 - eres2;
    p1 = std::exp(2.0 * (std::sqrt(level_dens * std::fabs(s.exc_energy - egam)) - xsqr)) *
         gammaR2 * gammaE2 / (egdp2 * egdp2 + gammaR2);
    s.probability += (p1 + p0);
    s.cumm[i] = s.probability;
    p0 = p1;
  }

  // NormC = 1.25 mb / (pi^2 hbarc^2). The factor A is the number of nucleons the resonance is
  // built on; the 0.5 the trapezoid rule would want is folded into the 1.25.
  const double NormC = 1.25 * millibarn() / (pi2() * u::hbarc<double>() * u::hbarc<double>());
  s.probability *= s.step * NormC * frag.a;
  return s.probability;
}

/// G4GammaTransition::SampleTransition, with polarFlag false so the direction is isotropic.
///
/// `nucleus` is modified into the residual through SetExcEnergyAndMomentum, and the emitted
/// gamma or electron is returned as a Fragment with A = 0 and a PDG code. The two-body
/// kinematics are done in the rest frame and boosted, and `ecm` is raised to `mass + emass`
/// when the fragment does not have the energy - which conserves momentum by giving both
/// products zero of it rather than by producing a negative square root.
template <typename Rng>
__host__ __device__ inline Fragment sample_gamma_transition_kinematics(
    Fragment& nucleus, double new_exc_energy, bool is_gamma, Rng& rng) {
  // bond_energy is identically zero in this configuration: it is non-zero only when a shell
  // index came back from G4NucLevel::SampleShell, and with fStoreAllLevels = false there is no
  // shell-probability table to sample, so the index is always -1. G4AtomicShells is therefore
  // never reached.
  const double bond_energy = 0.0;

  LorentzVector lv = nucleus.momentum;
  const double mass = nucleus.ground_state_mass + new_exc_energy;
  const double emass = is_gamma ? 0.0 : u::electron_mass_c2<double>();

  const Vec3d dir = random_direction(rng);

  double ecm = lv.mag();
  const Vec3d bst = lv.boost_vector();
  if (!is_gamma) { ecm += (u::electron_mass_c2<double>() - bond_energy); }
  if (ecm < mass + emass) { ecm = mass + emass; }
  double energy = 0.5 * ((ecm - mass) * (ecm + mass) + emass * emass) / ecm;
  const double mom = (emass > 0.0) ? std::sqrt((energy - emass) * (energy + emass)) : energy;

  LorentzVector res4mom(dir * mom, energy);
  double res_e = ecm - energy;
  if (res_e < mass) { res_e = mass; }
  lv = LorentzVector(dir * (-mom), res_e);
  lv.boost(bst);
  nucleus.set_exc_energy_and_momentum(new_exc_energy, lv);

  res4mom.boost(bst);
  Fragment out;
  out.z = 0;
  out.a = 0;
  out.momentum = res4mom;
  out.ground_state_mass = emass;
  out.excitation = 0.0;
  out.pdg_if_not_nucleus = is_gamma ? kPdgGamma : kPdgElectron;
  return out;
}

/// What one call of G4PhotonEvaporation::GenerateGamma produced.
struct GammaEmission {
  bool emitted = false;
  Fragment product;
  /// Set when the transition would have needed the correlated-gamma path, which is refused.
  bool refused_polarization = false;
};

/// G4PhotonEvaporation::GenerateGamma. Returns `emitted = false` when the cascade has reached a
/// state it will not leave - the ground state, a level with a lifetime above 1 ns, or an
/// excitation below tolerance - and modifies `nucleus` into the residual otherwise.
///
/// The three branches, in Geant4's order:
///   * a DISCRETE initial state (the excitation matches a tabulated level to within 10 eV and
///     that level has transitions): sample one transition, decide gamma versus internal
///     conversion from its `1/(1+alpha)`, sample the level's lifetime, and go to the final
///     level named by the transition.
///   * the ground state: return nothing, and mark the fragment long-lived if level 0's
///     lifetime is negative or above the 1 ns limit.
///   * CONTINUUM: sample a final energy from the cumulative array
///     GetEmissionProbability built, then snap it to the nearest tabulated level below the
///     current excitation.
///
/// One subtlety that is easy to lose: a level with a FLOATING marker and no transitions falls
/// back to the level below it when the two are within tolerance, because the floating level is
/// a duplicate energy carrying only a different marker.
///
/// @param creation_time  where the sampled level lifetime is accumulated, or null to discard
///        it. `G4PhotonEvaporation::GenerateGamma` reads `nucleus->GetCreationTime()`, adds
///        `-ltime*G4Log(G4UniformRand())` to it when the level is not prompt, and writes the
///        result onto BOTH the emitted gamma and the residual - so the value is a running
///        total down a cascade and not a per-transition delta. This module has no creation
///        time on a Fragment and discarded it; `G4NeutronRadCapture::ApplyYourself` is the
///        first caller that needs it, because it gives every secondary
///        `time + max(f->GetCreationTime(), 0.0)`. Null keeps the previous behaviour exactly:
///        the draw is still made, so the random stream is unchanged either way.
template <typename Rng>
__host__ __device__ inline GammaEmission generate_gamma(PhotonEvaporationState& s,
                                                        Fragment& nucleus,
                                                        const data::LevelTable& lt, Rng& rng,
                                                        double* creation_time = nullptr) {
  GammaEmission out;
  const double tolerance = deex_params().min_excitation;
  const double eexc = nucleus.excitation;
  if (eexc <= tolerance) { return out; }

  initialise_level_manager(s, lt, nucleus.z, nucleus.a);
  nucleus.long_lived = false;

  double efinal = 0.0;
  bool is_gamma = true;
  bool is_discrete = false;
  int ntrans = 0;
  const int m = s.man;

  if (m >= 0 && eexc <= s.level_energy_max + tolerance) {
    s.level_index = data::nearest_level_index(lt, m, eexc);
    const double elevel = data::level_energy(lt, m, s.level_index);
    is_discrete = (std::fabs(elevel - eexc) < tolerance);
    if (is_discrete && s.level_index > 0) {
      ntrans = data::level_ntrans(lt, m, s.level_index);
      if (data::level_floating(lt, m, s.level_index) > 0 && ntrans == 0 &&
          std::fabs(elevel - data::level_energy(lt, m, s.level_index - 1)) < tolerance) {
        const int prev = s.level_index - 1;
        if (data::level_ntrans(lt, m, prev) > 0) {
          s.level_index = prev;
          ntrans = data::level_ntrans(lt, m, prev);
        }
      }
    }
    if (ntrans == 0) { is_discrete = false; }
  }

  if (!is_discrete) {
    if (s.code != 1000 * nucleus.z + nucleus.a || eexc != s.exc_energy) {
      photon_emission_probability(s, nucleus, lt);
    }
    if (s.probability == 0.0) {
      s.points = 1;
      efinal = 0.0;
    } else {
      const double y = s.cumm[s.points - 1] * rng.uniform();
      for (int i = 1; i < s.points; ++i) {
        if (y <= s.cumm[i]) {
          efinal = s.step * ((i - 1) + (y - s.cumm[i - 1]) / (s.cumm[i] - s.cumm[i - 1]));
          break;
        }
      }
    }
    if (m >= 0) {
      if (efinal < s.level_energy_max) {
        s.level_index = data::nearest_level_index(lt, m, efinal, s.level_index);
        efinal = data::level_energy(lt, m, s.level_index);
        if (efinal >= eexc && s.level_index > 0) {
          --s.level_index;
          efinal = data::level_energy(lt, m, s.level_index);
        }
        nucleus.floating_level = data::level_floating(lt, m, s.level_index);
      } else {
        efinal = s.level_energy_max;
        s.level_index = data::nearest_level_index(lt, m, efinal, s.level_index);
      }
    }
  } else if (s.level_index == 0) {
    bool is_ll = false;
    if (m >= 0) {
      const double ltime = data::level_lifetime(lt, m, 0);
      if (ltime < 0.0 || ltime > deex_params().max_life_time) { is_ll = true; }
    }
    nucleus.long_lived = is_ll;
    return out;
  } else {
    const double ltime = data::level_lifetime(lt, m, s.level_index);
    // A negative lifetime means stable; above fMaxLifeTime (1 ns) the fragment is an isomer and
    // the cascade stops here, which is what fIsomerFlag being on means in practice.
    if (ltime < 0.0 || ltime > deex_params().max_life_time) {
      nucleus.long_lived = true;
      return out;
    }
    int idx = 0;
    if (ntrans > 1) { idx = data::sample_gamma_transition(lt, m, s.level_index, rng.uniform()); }
    const data::LevelTransition& tr = data::level_transition(lt, m, s.level_index, idx);
    const double prob = static_cast<double>(tr.prob);
    if (deex_params().internal_conversion && prob < 1.0) {
      const double rndm = rng.uniform();
      if (rndm > prob) {
        is_gamma = false;
        // The re-normalised random number would pick a shell here; with no shell table
        // G4NucLevel::SampleShell returns -1 and the draw is consumed and discarded. Consumed
        // deliberately: it is part of the stream Geant4 uses.
        (void)((rndm - prob) / (1.0 - prob));
      }
    }
    // With fCorrelatedGamma false the multipolarity ratio and the two spins are computed and
    // then not used - SampleDirection takes the isotropic branch. Kept as a comment rather
    // than as dead locals.
    s.level_index = data::transition_final_index(tr);
    efinal = data::level_energy(lt, m, s.level_index);
    nucleus.floating_level = data::level_floating(lt, m, s.level_index);
    // The level lifetime is sampled because fSampleTime is !fRDM = true. A Fragment carries no
    // creation time here, so the sample went into `creation_time` if the caller offered one and
    // was discarded otherwise - the draw is made either way, which is what keeps the random
    // stream aligned with Geant4's. This USED to be discarded unconditionally, with a comment
    // saying a neutron time cut would need it: P7's G4NeutronRadCapture is that caller, and it
    // gives every secondary `time + max(f->GetCreationTime(), 0.0)`.
    //
    // `time -= ltime*G4Log(G4UniformRand())` with the log of a uniform in (0,1) being negative,
    // so the delay ADDS. Accumulated, not assigned: the value is the running total down the
    // cascade, and Geant4 gets that by reading it off the residual (`nucleus->GetCreationTime()`
    // at the top of this function) and writing it back at the bottom.
    if (ltime > 0.0) {
      const double delay = -ltime * std::log(rng.uniform());
      if (creation_time != nullptr) { *creation_time += delay; }
    }
  }

  bool is_ll = false;
  if (m >= 0) {
    const double ltime = data::level_lifetime(lt, m, s.level_index);
    if (ltime < 0.0 || ltime > deex_params().max_life_time) { is_ll = true; }
  }
  nucleus.long_lived = is_ll;

  // A floating level at the same energy as the current one: nothing to emit.
  if (std::fabs(efinal - eexc) <= tolerance) { return out; }

  out.product = sample_gamma_transition_kinematics(nucleus, efinal, is_gamma, rng);
  out.emitted = true;

  // The ground state reached through a floating level with zero energy.
  if (efinal == 0.0 && s.level_index > 0) {
    s.level_index = 0;
    if (m >= 0) { nucleus.floating_level = data::level_floating(lt, m, 0); }
  }
  return out;
}

/// The correlated-gamma path: G4NuclearPolarization, G4NuclearPolarizationStore and
/// G4PolarizationTransition's Clebsch-Gordan angular correlations, reached when
/// fCorrelatedGamma is true and the initial spin is at or below fTwoJMAX.
///
/// **REFUSED, by name.** fCorrelatedGamma is false in 11.1.1 and no QBBC constructor turns it
/// on, so the direction of every discrete gamma in this configuration is isotropic. A caller
/// that sets correlated_gamma has the handler report this instead of getting isotropic gammas
/// under a name that promises correlated ones.
__host__ __device__ inline const char* refused_correlated_gamma() {
  return "G4PolarizationTransition / G4NuclearPolarization (correlated-gamma angular "
         "distribution, fCorrelatedGamma)";
}

}  // namespace g4gpu::deex

#endif
