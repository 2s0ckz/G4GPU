// G4ExcitedStringDecay::FragmentStrings: a list of excited strings in, one list of hadrons out,
// with the energy-momentum correction loop that makes the sum come back.
//
// Transcribed from
//   source/processes/hadronic/models/parton_string/hadronization/src/G4ExcitedStringDecay.cc
//     FragmentStrings, FragmentString, EnergyAndMomentumCorrector
//   source/processes/hadronic/util/include/G4ExcitedString.hh
//     Get4Momentum, LorentzRotate, IsExcited
//   source/processes/hadronic/util/src/G4SampleResonance.cc + .hh
//     SampleMass(pole, gamma, min, max), BrWigInt0, BrWigInv
//
// WHAT THIS LAYER ADDS to lund_fragment.cuh, and it is not bookkeeping:
//
//  1. EVERY SHORT-LIVED PRODUCT HAS ITS MASS RESAMPLED. A rho+ leaves the Lund fragmentation
//     with the PDG pole mass; FragmentStrings replaces it with a Breit-Wigner deviate between
//     `GetMinimumMass(def) + 10 MeV` and `pole + 5*width`, KEEPING THE THREE-MOMENTUM and
//     recomputing the energy. So the fragmentation's own energy-momentum balance is broken
//     here, deliberately, and item 2 is what repairs it. Every such resample spends one uniform
//     deviate, and the resonances are most of what a string makes - the rho, the omega, the
//     K*, the Delta - so a port that skips this gets both the masses and the whole remaining
//     random stream wrong.
//
//  2. THE CORRECTOR IS NOT A SAFETY NET, IT RUNS ON ALMOST EVERY EVENT. `NeedEnergyCorrector`
//     is set when any string's hadrons differ from the string's own energy by more than
//     `perMillion`, which after the resampling in item 1 is the normal case rather than the
//     exception. It rescales every hadron's three-momentum by a common factor in the total
//     c.m.s., iterating until the energies sum to the collision mass, at most 500 times.
//
//  3. The whole thing runs in the c.m.s. of the STRINGS, and the strings are mutated into that
//     frame on the way in. On failure - `success` false after 100 attempts - Geant4 rotates the
//     strings BACK to the lab and returns a null vector; the port does the same and reports
//     `kEnergyCorrectorFailed`, because a null vector reaching G4VPartonStringModel::Scatter is
//     a silent retry of the whole interaction.
//
// THE NOT-EXCITED STRING. `G4ExcitedString::IsExcited()` is `theTrack == 0`: a string that was
// never excited carries a HADRON, not a parton pair, and FragmentStrings copies it into the
// result through the G4KineticTrack constructor - which means it goes through the kaon0 coin
// toss and spends a deviate, exactly as a fragmentation product does. See
// ftf_kinetic_track_pdg in lund_fragment.cuh.
#pragma once
#include <cfloat>
#include <cmath>

#include "core/units.cuh"
#include "data/ftf_hadrons.hh"
#include "physics/hadronic/ftf/lorentz.cuh"
#include "physics/hadronic/ftf/lund_fragment.cuh"
#include "physics/hadronic/ftf/refusal.cuh"

namespace g4gpu::hadronic::ftf {

/// G4ExcitedString as the string model hands it over. TWO partons: the kinky-string arm that
/// would add a gluon is refused by name (`kKinkyStrings`, Pt2Kink = 0 in the G4FTFParameters
/// constructor), so `Get4Momentum()`'s sum over thePartons is `pleft + pright`.
struct ExcitedString {
  int left = 0;    ///< thePartons.front(), the LEFT parton's PDG code
  int right = 0;   ///< thePartons.back()
  Vec4 pleft;
  Vec4 pright;
  int direction = 1;  ///< PROJECTILE(+1) or TARGET(-1); G4ExcitedString requires 1 or -1
  double time_of_creation = 0.0;
  Vec3d position{0.0, 0.0, 0.0};

  /// theTrack != 0, i.e. IsExcited() is false and the string is one hadron already.
  bool excited = true;
  int track_pdg = 0;
  Vec4 track_mom;
  double track_time = 0.0;
  Vec3d track_position{0.0, 0.0, 0.0};
};

/// G4ExcitedString::Get4Momentum.
__host__ __device__ inline Vec4 ftf_string_4momentum(const ExcitedString& s) {
  return s.excited ? (s.pleft + s.pright) : s.track_mom;
}

/// The state one interaction's string decay needs. `frag` is the per-string workspace, so a
/// caller holds ONE of these and not one per string: 33,320 bytes for it plus 256 hadrons at
/// 72 bytes plus their saved masses, about 54 kB, and every byte of it behind a pointer.
template <typename real_t, int kMaxOut = 256, int kPerString = 96>
struct StringsWorkspace {
  static constexpr int kMaxHadrons = kMaxOut;

  FragmentWorkspace<real_t, kPerString> frag;

  FragHadron out[kMaxOut];
  /// EnergyAndMomentumCorrector's `HadronMass` vector: the invariant mass each hadron had when
  /// it entered the corrector, which after the resonance resampling is NOT its PDG mass. The
  /// corrector scales momenta and recomputes energies from THESE, so they have to be kept.
  real_t mass[kMaxOut];
  int n_out = 0;

  int attempts = 0;            ///< FragmentStrings' `attempts`, for the report
  bool need_corrector = false;
  bool corrector_ran = false;
  bool success = false;
  FtfRefusal refused = FtfRefusal::kNone;
};

/// G4SampleResonance::BrWigInt0 and BrWigInv - the Breit-Wigner CDF and its inverse.
template <typename real_t>
__host__ __device__ inline real_t ftf_brwig_int0(real_t x, real_t gamma, real_t m0) {
  return real_t(2.0) * gamma * std::atan(real_t(2.0) * (x - m0) / gamma);
}
template <typename real_t>
__host__ __device__ inline real_t ftf_brwig_inv(real_t x, real_t gamma, real_t m0) {
  return real_t(0.5) * gamma * std::tan(real_t(0.5) * x / gamma) + m0;
}

/// G4SampleResonance::SampleMass(poleMass, gamma, minMass, maxMass).
///
/// The `minMass > maxMass` arm is A.R.'s 2017 protection for a wide daughter of a wide parent;
/// it replaces the minimum by the maximum rather than throwing. The zero-width arm spends NO
/// deviate, which matters for the draw count: a stable hadron never reaches this function at
/// all (only `IsShortLived()` products do), but a short-lived one with zero width would.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t ftf_sample_resonance_mass(real_t pole_mass, real_t gamma,
                                                            real_t min_mass, real_t max_mass,
                                                            Rng& rng) {
  real_t protected_min = min_mass;
  if (min_mass > max_mass) { protected_min = max_mass; }

  if (gamma < static_cast<real_t>(DBL_EPSILON)) {
    const real_t hi = (max_mass < pole_mass) ? max_mass : pole_mass;
    return (min_mass > hi) ? min_mass : hi;
  }
  const real_t fmin = ftf_brwig_int0(protected_min, gamma, pole_mass);
  const real_t fmax = ftf_brwig_int0(max_mass, gamma, pole_mass);
  const real_t f = fmin + (fmax - fmin) * static_cast<real_t>(rng.uniform());
  return ftf_brwig_inv(f, gamma, pole_mass);
}

/// G4ExcitedStringDecay::EnergyAndMomentumCorrector.
///
/// Returns false the four ways Geant4 returns FALSE - fewer than two hadrons, a hadron mass sum
/// already above the collision mass, a spacelike hadron sum, and 500 iterations without
/// |Scale - 1| <= 1e-5 - and TRUE for an empty list, which is Geant4's first line and is not a
/// success in any physical sense.
///
/// Note which boost it uses: `Beta = -TotalCollisionMom.boostVector()`, with the commented-out
/// alternative `-SumMom.boostVector()` left in the original. The two differ once the hadrons do
/// not carry the string's momentum, which is exactly when the corrector is called.
template <typename real_t, int kMaxOut, int kPerString>
__host__ __device__ inline bool ftf_energy_momentum_corrector(
    StringsWorkspace<real_t, kMaxOut, kPerString>* ws, const Vec4& total_collision_mom) {
  const int n_attempt_scale = 500;
  const real_t err_limit = real_t(1.0e-5);
  if (ws->n_out == 0) { return true; }

  Vec4 sum_mom(0.0, 0.0, 0.0, 0.0);
  real_t sum_mass = real_t(0);
  const real_t total_collision_mass = static_cast<real_t>(total_collision_mom.mag());

  for (int i = 0; i < ws->n_out; ++i) {
    sum_mom = sum_mom + ws->out[i].momentum;
    ws->mass[i] = static_cast<real_t>(ws->out[i].momentum.mag());
    sum_mass += ws->mass[i];
  }

  if (ws->n_out < 2) { return false; }
  if (sum_mass > total_collision_mass) { return false; }
  // `SumMass = SumMom.m2()` - the SIGNED invariant, reused as a scratch variable. The guard
  // below it is DEAD, and measured so rather than argued: removing it changes nothing in any
  // oracle row, because the sum of timelike future-pointing four-vectors is timelike whatever
  // the momenta were sampled to be. A hadron list cannot produce a negative m2 - each term has
  // e > |p| and the property survives addition - so no input reaches the `return false`.
  // Transcribed because it is in the source and because a future caller that hands the
  // corrector something other than hadrons would need it.
  const real_t sum_m2 = static_cast<real_t>(sum_mom.e * sum_mom.e - g4gpu::mag2(sum_mom.v));
  if (sum_m2 < real_t(0)) { return false; }

  const Vec3d bv = total_collision_mom.boost_vector();
  const Vec3d beta_in{-bv.x, -bv.y, -bv.z};
  for (int i = 0; i < ws->n_out; ++i) { ws->out[i].momentum.boost(beta_in); }

  real_t scale = real_t(1);
  bool success = false;
  for (int attempt = 0; attempt < n_attempt_scale; ++attempt) {
    real_t sum = real_t(0);
    for (int i = 0; i < ws->n_out; ++i) {
      const real_t m = ws->mass[i];
      Vec3d v = ws->out[i].momentum.v;
      v.x *= static_cast<double>(scale);
      v.y *= static_cast<double>(scale);
      v.z *= static_cast<double>(scale);
      const double e = std::sqrt(g4gpu::mag2(v) + static_cast<double>(m) * static_cast<double>(m));
      ws->out[i].momentum = Vec4(v, e);
      sum += static_cast<real_t>(e);
    }
    scale = total_collision_mass / sum;
    if (std::fabs(scale - real_t(1)) <= err_limit) {
      success = true;
      break;
    }
  }

  const Vec3d beta_out = total_collision_mom.boost_vector();
  for (int i = 0; i < ws->n_out; ++i) { ws->out[i].momentum.boost(beta_out); }
  return success;
}

/// G4ExcitedStringDecay::FragmentStrings.
///
/// `strings` is mutated, as Geant4 mutates `theStrings`: the partons are transformed into the
/// strings' c.m.s. on the way in and, ONLY IF THE WHOLE THING FAILED, back to the lab at the
/// end. A caller that retries therefore hands the next attempt strings in the lab frame after a
/// failure and strings in the c.m.s. after a success - which is Geant4's behaviour and is why
/// G4VPartonStringModel::Scatter rebuilds the strings rather than reusing them.
template <typename real_t, int kMaxOut, int kPerString, typename Rng>
__host__ __device__ inline void ftf_fragment_strings(
    const LundTables<real_t>* t, StringsWorkspace<real_t, kMaxOut, kPerString>* ws,
    ExcitedString* strings, int n_strings, Rng& rng) {
  ws->n_out = 0;
  ws->attempts = 0;
  ws->need_corrector = false;
  ws->corrector_ran = false;
  ws->success = false;
  ws->refused = FtfRefusal::kNone;

  // The total four-momentum, in the lab, of everything that will be fragmented.
  Vec4 kt_sum(0.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < n_strings; ++i) { kt_sum = kt_sum + ftf_string_4momentum(strings[i]); }

  const Vec3d bv = kt_sum.boost_vector();
  const LorentzRot to_cms = lorentz_boost(Vec3d{-bv.x, -bv.y, -bv.z});
  const LorentzRot to_lab = lorentz_inverse(to_cms);

  // Into the c.m.s., and KTsum is REBUILT there from the transformed partons rather than
  // transformed itself - the two differ in the last bits, and this one is what the corrector
  // is given.
  kt_sum = Vec4(0.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < n_strings; ++i) {
    if (strings[i].excited) {
      strings[i].pleft = lorentz_apply(to_cms, strings[i].pleft);
      strings[i].pright = lorentz_apply(to_cms, strings[i].pright);
    } else {
      strings[i].track_mom = lorentz_apply(to_cms, strings[i].track_mom);
    }
    kt_sum = kt_sum + ftf_string_4momentum(strings[i]);
  }

  const int max_attempts = 100;
  bool success = false;
  bool need_corrector = false;
  while (!success && ws->attempts < max_attempts) {
    ws->n_out = 0;
    ++ws->attempts;
    need_corrector = false;
    // The refusal is cleared with the hadron list, and for the same reason. A string that ran
    // Loop_toFragmentString out of attempts leaves `kFragmentLoopExhausted` behind, and
    // FragmentStrings' answer to that is to try the whole set again - so a refusal from a
    // DISCARDED attempt is not a refusal of the result. What survives is what the attempt that
    // produced the returned hadrons reported.
    ws->refused = FtfRefusal::kNone;

    for (int i = 0; i < n_strings; ++i) {
      const int first = ws->n_out;

      if (strings[i].excited) {
        ftf_fragment_string(t, &ws->frag, strings[i].left, strings[i].right, strings[i].pleft,
                            strings[i].pright, strings[i].direction, strings[i].time_of_creation,
                            strings[i].position, rng);
        if (ws->frag.refused != FtfRefusal::kNone) { ws->refused = ws->frag.refused; }
        for (int h = 0; h < ws->frag.n_out; ++h) {
          if (ws->n_out >= kMaxOut) {
            ws->refused = FtfRefusal::kStringHadronCapacity;
            return;
          }
          ws->out[ws->n_out++] = ws->frag.out[h];
        }
      } else {
        // `new G4KineticTrack(definition, formationTime, G4ThreeVector(0), Mom)` - the
        // constructor, so the kaon0 coin toss happens here too, and then SetPosition.
        if (ws->n_out >= kMaxOut) {
          ws->refused = FtfRefusal::kStringHadronCapacity;
          return;
        }
        FragHadron h;
        h.pdg = ftf_kinetic_track_pdg<real_t>(strings[i].track_pdg, rng);
        h.momentum = strings[i].track_mom;
        h.formation_time = strings[i].track_time;
        h.position = strings[i].track_position;
        ws->out[ws->n_out++] = h;
      }

      // "No KineticTracks produced" - Geant4 `continue`s, which also skips the `success = true`
      // at the bottom of the loop body. So a single string that produces nothing does not by
      // itself force a retry, but a LAST string that produces nothing leaves success at
      // whatever the previous string set.
      if (ws->n_out == first) { continue; }

      Vec4 kt_sum1(0.0, 0.0, 0.0, 0.0);
      for (int h = first; h < ws->n_out; ++h) {
        const data::FtfHadron* d = data::ftf_find_hadron(ws->out[h].pdg);
        if (d == nullptr) {
          ws->refused = FtfRefusal::kUnknownHadronCode;
        } else if (d->shortlived) {
          if (d->minmass < 0.0) {
            // G4SampleResonance::GetMinimumMass dereferences GetDecayTable() with no null
            // check, so Geant4 would crash here; ref/oracle/ftf_hadrons.csv writes -1 for the
            // short-lived particles that have no decay table (the diquarks). Unreachable from
            // a fragmentation product, and reported rather than guessed.
            ws->refused = FtfRefusal::kResonanceMinimumMassMissing;
          } else {
            const real_t new_mass = ftf_sample_resonance_mass<real_t>(
                static_cast<real_t>(d->mass), static_cast<real_t>(d->width),
                static_cast<real_t>(d->minmass) + real_t(10.0) * units::MeV<real_t>(),
                static_cast<real_t>(d->mass) + real_t(5.0) * static_cast<real_t>(d->width), rng);
            const Vec3d p3 = ws->out[h].momentum.v;
            ws->out[h].momentum =
                Vec4(p3, std::sqrt(g4gpu::mag2(p3) + static_cast<double>(new_mass) *
                                                         static_cast<double>(new_mass)));
          }
        }
        kt_sum1 = kt_sum1 + ws->out[h].momentum;
      }

      // CLHEP's `perMillion`, 1e-6 (src/g4/G4SystemOfUnits.hh), written out rather than taken
      // from the decay module's copy so that this file depends on nothing outside ftf/.
      const double string_e = ftf_string_4momentum(strings[i]).e;
      if (kt_sum1.e > 0.0 && std::fabs((kt_sum1.e - string_e) / kt_sum1.e) > 1.0e-6) {
        need_corrector = true;
      }
      success = true;
    }

    if (need_corrector) { success = ftf_energy_momentum_corrector(ws, kt_sum); }
  }
  ws->need_corrector = need_corrector;
  ws->corrector_ran = need_corrector;
  ws->success = success;

  for (int h = 0; h < ws->n_out; ++h) {
    ws->out[h].momentum = lorentz_apply(to_lab, ws->out[h].momentum);
  }

  if (!success) {
    // Geant4 deletes the result and returns a NULL vector, and puts the strings back in the
    // lab frame so that the caller can try again with them.
    ws->n_out = 0;
    ws->refused = FtfRefusal::kEnergyCorrectorFailed;
    for (int i = 0; i < n_strings; ++i) {
      if (strings[i].excited) {
        strings[i].pleft = lorentz_apply(to_lab, strings[i].pleft);
        strings[i].pright = lorentz_apply(to_lab, strings[i].pright);
      } else {
        strings[i].track_mom = lorentz_apply(to_lab, strings[i].track_mom);
      }
    }
  }
}

}  // namespace g4gpu::hadronic::ftf
