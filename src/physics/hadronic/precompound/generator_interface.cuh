// How a cascade's residual becomes the fragment G4PreCompoundModel::DeExcite receives.
//
// Transcribed from G4GeneratorPrecompoundInterface (models/binary_cascade) in 11.1.1:
// Propagate, PropagateNuclNucl and MakeCoalescence.
//
// Every string and cascade model in QBBC ends here. It takes a list of kinetic tracks and a
// wounded nucleus and produces (a) the tracks that escaped, as final-state secondaries, and
// (b) one or two excited fragments with an exciton configuration, which go to DeExcite. The
// plain data types below are the contract P9 (binary cascade), P10 (Bertini) and P11 (FTFP)
// fill in; they exist here because this is the file that reads them.
//
// **Three constants, from the constructor**: CaptureThreshold = 70 MeV, DeltaM = 5.0 MeV,
// DeltaR = 0.0. DeltaR is set and never read - the space cut it belongs to is commented out
// inside MakeCoalescence - so it is recorded and not ported.
//
// **The capture test is an exponential, not a threshold.** `e - mass > -CaptureThreshold *
// G4Log(G4UniformRand())` compares the track's kinetic energy against an Exp(70 MeV) deviate,
// so a nucleon inside the nuclear radius is CAPTURED with probability exp(-T/70 MeV) and
// escapes otherwise. The name says threshold and the code samples. Note also that `mass` is
// `Get4Momentum().mag()`, the track's own invariant mass, not the PDG mass - so an off-shell
// track's kinetic energy is measured against its own mass shell.
//
// **A residual with Z > A is dropped, silently, in Propagate.** The whole de-excitation block
// is inside `if (anA >= aZ)`, with no else - so a residual whose charge exceeds its mass
// number contributes nothing to the final state at all and its energy leaves the event. The
// port reports it (`GeneratorRefusal::dropped_residual`) rather than reproducing the silence,
// because the only way anyone finds out is a printed number - the same reasoning
// docs/PORTED.md 2.2 gives for the neutron killer's discarded kinetic energy.
//
// **REFUSED, by name:**
//   * `G4DecayKineticTracks` - the first line of both Propagate entry points decays every
//     short-lived track in the list through `G4KineticTrack::Decay()`, which is the cascade's
//     own resonance machinery (a decay table sampled at the track's off-shell mass) and not
//     this package's. `CascadeTrack::is_short_lived` carries the fact, and a list containing
//     one is refused by name instead of being de-excited as though the resonance were a
//     stable secondary.
//   * the anti-nucleus branches of PropagateNuclNucl - `ProjectileIsAntiNucleus`, the
//     `aZb += charge - 0.1` sign flip, and the fourteen-way SetDefinitionAndUpdateE
//     substitution to anti-species at the end. Anti-nuclei are P1's refused set.
//   * the hypernucleus branch - `aLb > 0`, `G4HyperNucleiProperties::GetNuclearMass` and the
//     six anti-hyper substitutions. P3 refuses `nL != 0` in the handler for the same reason.
//   * `GetPrimaryProjectile()` in the QGS branch of Propagate needs the primary's four
//     momentum, which the interface reads off G4HadronicInteraction. It is a PARAMETER here
//     rather than a refusal, because a caller that has it can pass it; a caller that leaves
//     it zero and hits the QGS branch is refused.
#ifndef G4GPU_PRECO_GENERATOR_INTERFACE_CUH
#define G4GPU_PRECO_GENERATOR_INTERFACE_CUH

#include <cmath>

#include "core/units.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/precompound/precompound_model.cuh"

namespace g4gpu::preco {

namespace u = g4gpu::units;
using deex::Fragment;
using deex::LorentzVector;
using deex::Vec3d;

/// G4GeneratorPrecompoundInterface's three constructor constants.
__host__ __device__ inline constexpr double capture_threshold() {
  return 70.0 * u::MeV<double>();
}
__host__ __device__ inline constexpr double coalescence_delta_m() {
  return 5.0 * u::MeV<double>();
}
/// DeltaR = 0.0, and it is never read: MakeCoalescence's space cut is commented out. Kept as a
/// named zero so that a future release which uncomments the cut is a compile-time question.
__host__ __device__ inline constexpr double coalescence_delta_r() { return 0.0; }

/// One entry of the G4KineticTrackVector a cascade hands over.
///
/// Only what Propagate and MakeCoalescence read: the definition (as a PDG code and a charge),
/// the four-momentum, the formation position and time, and the creator-model id it copies
/// through. `is_short_lived` is `GetDefinition()->IsShortLived()`, which decides whether
/// G4DecayKineticTracks would have decayed it - see the refusal in the file header.
struct CascadeTrack {
  int pdg = 0;
  int charge = 0;                ///< in units of eplus, already an integer
  LorentzVector momentum;
  Vec3d position{0.0, 0.0, 0.0};
  double formation_time = 0.0;
  int creator_model_id = -1;
  bool is_short_lived = false;
};

/// One hit nucleon of the wounded nucleus, i.e. what the `AreYouHit()` loops read.
struct HitNucleon {
  int charge = 0;                ///< G4int(GetDefinition()->GetPDGCharge()/eplus + 0.1)
  double pdg_mass = 0.0;         ///< for the QGSM test below
  double binding_energy = 0.0;   ///< GetBindingEnergy, read only by PropagateNuclNucl
  LorentzVector momentum;
  bool is_lambda = false;        ///< GetParticleType() == G4Lambda - the hypernucleus path
};

/// The G4V3DNucleus the cascade leaves behind.
///
/// `a`/`z` are the INITIAL mass number and charge (GetMassNumber/GetCharge); the hit nucleons
/// are subtracted from them here, exactly as the interface's loops do. `radius` is
/// GetNuclearRadius.
struct WoundedNucleus {
  int a = 0;
  int z = 0;
  int lambdas = 0;
  double radius = 0.0;
  const HitNucleon* hit = nullptr;
  int n_hit = 0;
};

/// What the interface refused or had to report.
struct GeneratorRefusal {
  bool short_lived_track = false;   ///< G4DecayKineticTracks would have run
  bool anti_nucleus = false;        ///< the ProjectileIsAntiNucleus branches
  bool hyper_nucleus = false;       ///< aLb > 0
  bool dropped_residual = false;    ///< Propagate's `anA >= aZ` with no else
  bool missing_primary = false;     ///< the QGS branch needs GetPrimaryProjectile()
  bool capacity = false;
  int refused_z = 0, refused_a = 0;

  __host__ __device__ bool any() const {
    return short_lived_track || anti_nucleus || hyper_nucleus || dropped_residual ||
           missing_primary || capacity;
  }
};

/// The residual an interface built, before DeExcite sees it: the fragment plus its exciton
/// configuration. Returned separately from the de-excitation so that a test can compare the
/// hand-over itself - which is the part this package owns - without the cascade underneath it.
struct CascadeResidual {
  Fragment fragment;
  Excitons excitons;
  bool exists = false;      ///< false when anA == 0 or the `anA >= aZ` test failed
  bool qgsm = false;        ///< the wounded-nucleon test below chose the QGS branch
  LorentzVector escaped;    ///< Secondary4Momentum, the sum over the tracks that got out
  LorentzVector captured;   ///< captured4Momentum
  LorentzVector holes;      ///< Residual4Momentum, minus the sum of the hit nucleons
};

/// The `QGSM` detection loop, verbatim: the model that ran is taken to be QGS if ANY hit
/// nucleon is off its mass shell downwards, `Get4Momentum().mag() < GetPDGMass()`.
///
/// This is a second full pass over the wounded nucleus - the interface loops over it twice,
/// once to count holes and once to set this flag - and it is the only thing that distinguishes
/// the FTF branch from the QGS one. QBBC does not build a QGS model, so the branch it selects
/// is dead in this physics list; it is transcribed because the flag is cheap and because a
/// caller with off-shell nucleons would otherwise silently take the FTF arm.
__host__ __device__ inline bool wounded_nucleus_is_qgsm(const WoundedNucleus& nuc) {
  for (int i = 0; i < nuc.n_hit; ++i) {
    if (nuc.hit[i].momentum.mag() < nuc.hit[i].pdg_mass) { return true; }
  }
  return false;
}

/// G4GeneratorPrecompoundInterface::Propagate, up to but not including the DeExcite call.
///
/// `escaped` receives the tracks that leave, in the order Geant4 pushes them; `n_escaped` is
/// how many. A track escapes if it is not a nucleon, or if it formed outside the nuclear
/// radius, or if it wins the exponential capture test. Everything else is absorbed into the
/// residual: `++anA`, `++numberOfEx`, `aZ += Z`, `numberOfCh += Z`, and its four-momentum
/// added to `captured4Momentum`.
///
/// Then the wounded nucleus: each hit nucleon is one HOLE and one exciton, removes one unit of
/// A and its charge from Z, and SUBTRACTS its four-momentum from `Residual4Momentum` - which
/// starts at zero, so `holes` comes out with negative energy. That is not a bug: the sum
/// `Residual4Momentum + captured4Momentum` is the momentum the excited residual carries
/// RELATIVE to the untouched target, and Geant4's own comment says
/// "TargetNucleusMass is not need at the moment".
///
/// The mass fix-up after that sum is one-sided: if the invariant mass came out at or below the
/// residual's ground-state mass, the energy is raised to put it exactly on the mass shell -
/// giving E* = 0 - and if it came out above, nothing is done and E* is the excess. So a
/// residual can never be given a negative excitation, and `G4Fragment`'s own 10 eV floor
/// never fires from this path.
///
/// `n_particles = numberOfEx - numberOfHoles` is the count of captured nucleons: numberOfEx is
/// incremented once per capture AND once per hole, so the subtraction recovers the captures.
template <typename Rng>
__host__ __device__ inline CascadeResidual
propagate_residual(const CascadeTrack* tracks, int n_tracks, const WoundedNucleus& nuc,
                   const LorentzVector& primary_projectile, CascadeTrack* escaped,
                   int escaped_capacity, int& n_escaped, GeneratorRefusal& ref, Rng& rng) {
  CascadeResidual res;
  n_escaped = 0;

  for (int i = 0; i < n_tracks; ++i) {
    if (tracks[i].is_short_lived) {
      ref.short_lived_track = true;
      return res;
    }
  }
  if (nuc.lambdas != 0) {
    ref.hyper_nucleus = true;
    return res;
  }

  int anA = nuc.a;
  int aZ = nuc.z;
  int number_of_ex = 0;
  int number_of_ch = 0;
  int number_of_holes = 0;
  const double R = nuc.radius;

  LorentzVector captured;
  LorentzVector residual;    // starts at zero; the hit nucleons are subtracted from it
  LorentzVector secondary;

  for (int i = 0; i < n_tracks; ++i) {
    const CascadeTrack& t = tracks[i];
    const bool is_nucleon = (t.pdg == 2212 || t.pdg == 2112);
    const double e = t.momentum.e;
    const double mass = t.momentum.mag();
    bool escapes = false;
    if (!is_nucleon || g4gpu::mag(t.position) > R) {
      escapes = true;
    } else if (e - mass > -capture_threshold() * std::log(rng.uniform())) {
      escapes = true;
    }
    if (escapes) {
      if (n_escaped >= escaped_capacity) {
        ref.capacity = true;
        return res;
      }
      escaped[n_escaped++] = t;
      secondary += t.momentum;
    } else {
      ++anA;
      ++number_of_ex;
      const int Z = t.charge;
      aZ += Z;
      number_of_ch += Z;
      captured += t.momentum;
    }
  }

  for (int i = 0; i < nuc.n_hit; ++i) {
    ++number_of_holes;
    ++number_of_ex;
    --anA;
    aZ -= nuc.hit[i].charge;
    residual -= nuc.hit[i].momentum;
  }

  res.qgsm = wounded_nucleus_is_qgsm(nuc);
  res.escaped = secondary;
  res.captured = captured;
  res.holes = residual;

  if (anA == 0) { return res; }
  if (anA < aZ) {
    // Geant4's `if (anA >= aZ)` has no else: the residual disappears from the event.
    ref.dropped_residual = true;
    ref.refused_z = aZ;
    ref.refused_a = anA;
    return res;
  }

  const double fmass = deex::nuclear_mass(anA, aZ);
  LorentzVector exciton;
  if (!res.qgsm) {
    exciton = residual + captured;
    const double actual = exciton.mag();
    if (actual <= fmass) {
      exciton.e = std::sqrt(g4gpu::mag2(exciton.v) + fmass * fmass);
    }
  } else {
    if (primary_projectile.e == 0.0) {
      ref.missing_primary = true;
      return res;
    }
    const double initial_target_mass = deex::nuclear_mass(nuc.a, nuc.z);
    exciton = primary_projectile + LorentzVector(0.0, 0.0, 0.0, initial_target_mass) - secondary;
    const double actual = exciton.mag();
    if (actual - fmass < 0.0) {
      // Note the +10 MeV: the QGS branch does NOT put the residual on its mass shell, it puts
      // it 10 MeV above it, so the fragment arrives with E* = 10 MeV rather than zero.
      const double m = fmass + 10.0 * u::MeV<double>();
      exciton.e = std::sqrt(g4gpu::mag2(exciton.v) + m * m);
    }
  }

  res.fragment = deex::make_fragment(anA, aZ, exciton);
  res.excitons.particles = number_of_ex - number_of_holes;
  res.excitons.charged = number_of_ch;
  res.excitons.holes = number_of_holes;
  res.exists = true;
  return res;
}

// ---------------------------------------------------------------------------------------------
// MakeCoalescence
// ---------------------------------------------------------------------------------------------

/// G4GeneratorPrecompoundInterface::MakeCoalescence - replace close proton-neutron pairs by
/// deuterons. Called by PropagateNuclNucl only; Propagate does not call it.
///
/// The pair test is purely on invariant mass: `(p4 + n4).mag() <= m_d + 5 MeV`. The space cut
/// the method's own comment describes - `&& (EffDistance <= SpaceCut)` - is commented out in
/// 11.1.1, which makes two things dead: `DeltaR`, and the `NeutSPposition` four-vector built
/// two lines above with `formationTime*hbarc/fermi` in its time component (a unit conversion
/// applied to only one of the two positions, which would have made the cut asymmetric had it
/// been live). Neither is ported; both are recorded here.
///
/// The search is the outer loop over protons and, for each, the FIRST neutron that passes -
/// `break` after one match - and both are then removed. So it is greedy in list order and not
/// a minimum-mass pairing, which makes the result depend on the order the cascade pushed its
/// tracks. The deuteron is appended at the END of the list, with the mean formation time and
/// mean position of its two parents; because it is appended, the outer loop can reach it, but
/// it is not a proton so it is skipped.
///
/// Writes the surviving tracks into `out` in Geant4's resulting order: the un-paired tracks in
/// their original order, then the deuterons in the order they were created. Returns the count.
__host__ __device__ inline int make_coalescence(const CascadeTrack* in, int n_in,
                                               CascadeTrack* out, int out_capacity,
                                               GeneratorRefusal& ref) {
  const double mass_cut = deex::pdg_mass_deuteron() + coalescence_delta_m();
  int n_out = 0;
  // `consumed` is the null pointer Geant4 leaves in the vector before erasing it, and
  // `partner` records which neutron each proton took. Two passes over `in` rather than one
  // in-place edit, because the input is const and a kernel has nowhere to allocate. 64 is the
  // largest list this signature accepts and it is a stated refusal rather than a silent
  // truncation; a cascade at a few GeV makes tens of tracks, not hundreds.
  const int kMaxTracks = 64;
  if (n_in > kMaxTracks) {
    ref.capacity = true;
    return 0;
  }
  bool consumed[kMaxTracks];
  int partner[kMaxTracks];
  for (int i = 0; i < n_in; ++i) {
    consumed[i] = false;
    partner[i] = -1;
  }
  for (int i = 0; i < n_in; ++i) {
    if (consumed[i] || in[i].pdg != 2212) { continue; }
    for (int j = 0; j < n_in; ++j) {
      if (consumed[j] || in[j].pdg != 2112) { continue; }
      const LorentzVector sum = in[i].momentum + in[j].momentum;
      if (sum.mag() <= mass_cut) {
        consumed[i] = true;
        consumed[j] = true;
        partner[i] = j;
        break;
      }
    }
  }
  for (int i = 0; i < n_in; ++i) {
    if (consumed[i]) { continue; }
    if (n_out >= out_capacity) { ref.capacity = true; return n_out; }
    out[n_out++] = in[i];
  }
  for (int i = 0; i < n_in; ++i) {
    if (partner[i] < 0) { continue; }
    const int j = partner[i];
    if (n_out >= out_capacity) { ref.capacity = true; return n_out; }
    CascadeTrack d;
    d.pdg = 1000010020;
    d.charge = 1;
    d.momentum = in[i].momentum + in[j].momentum;
    d.formation_time = (in[i].formation_time + in[j].formation_time) / 2.0;
    d.position = 0.5 * (in[i].position + in[j].position);
    d.creator_model_id = -1;   // secID; the interface sets its own model id here
    out[n_out++] = d;
  }
  return n_out;
}

// ---------------------------------------------------------------------------------------------
// PropagateNuclNucl
// ---------------------------------------------------------------------------------------------

/// Both residuals PropagateNuclNucl builds. The projectile one is de-excited in ITS OWN rest
/// frame and the products boosted back, which the target one is not.
struct NuclNuclResiduals {
  CascadeResidual target;
  CascadeResidual projectile;
  Vec3d projectile_boost_to_cm{0.0, 0.0, 0.0};  ///< findBoostToCM of the projectile residual
};

/// G4GeneratorPrecompoundInterface::PropagateNuclNucl, the nucleus-on-nucleus form, up to but
/// not including the two DeExcite calls.
///
/// It differs from Propagate in five ways that are not cosmetic:
///
///   1. **The excitation energy is accumulated from binding energies**, not derived from the
///      four-momentum: `exEnergy += theCurrentNucleon->GetBindingEnergy()` over the hit
///      nucleons. It is then OVERWRITTEN by `RemnMass - fMass` whenever the four-momentum
///      already gives a mass above the ground state - so the binding-energy sum is used only
///      as a floor, and only when the momentum sum came out too light.
///   2. **The capture test is on the invariant-mass increase, not the kinetic energy**:
///      `-CaptureThreshold*log(rand()) > (track + residual).mag() - track.mag() -
///      residual.mag()`. That quantity is how much invariant mass absorbing the track would
///      add to the residual, which is the kinetic energy in the residual's rest frame - so it
///      is the same physics measured in the right frame, and it is NOT what Propagate does.
///   3. **Capture is gated on a remnant existing at all**: `ExistTargetRemnant` is
///      `numberOfHoles < 0.3*(numberOfHoles + anA)`, i.e. fewer than 30% of the nucleons were
///      hit. A nucleus that lost a third of itself captures nothing.
///   4. **A track that both nuclei could capture is assigned by a coin flip**, consuming a
///      third uniform deviate for that track.
///   5. **The projectile's capture radius test is in the projectile's frame**: the track's
///      (position, formationTime) four-vector is boosted by the primary projectile's velocity
///      before its magnitude is compared to Rb.
///
/// `MakeCoalescence` runs first, so the track list this sees may contain deuterons that were
/// a proton and a neutron a moment ago - and a deuteron is not a nucleon, so it always
/// escapes. Coalescence therefore REMOVES capture candidates.
template <typename Rng>
__host__ __device__ inline NuclNuclResiduals
propagate_nucl_nucl_residuals(const CascadeTrack* tracks, int n_tracks,
                              const WoundedNucleus& target, const WoundedNucleus& projectile,
                              const LorentzVector& primary_projectile,
                              int primary_baryon_number, CascadeTrack* escaped,
                              int escaped_capacity, int& n_escaped, GeneratorRefusal& ref,
                              Rng& rng) {
  NuclNuclResiduals out;
  n_escaped = 0;

  if (primary_baryon_number < -1) {
    ref.anti_nucleus = true;
    return out;
  }
  if (target.lambdas != 0 || projectile.lambdas != 0) {
    ref.hyper_nucleus = true;
    return out;
  }
  for (int i = 0; i < n_tracks; ++i) {
    if (tracks[i].is_short_lived) {
      ref.short_lived_track = true;
      return out;
    }
  }

  // The target's wounded loop, with the binding-energy sum Propagate does not have.
  int anA = target.a;
  int aZ = target.z;
  int n_ex = 0, n_ch = 0, n_holes = 0;
  double ex_energy = 0.0;
  LorentzVector target4;
  for (int i = 0; i < target.n_hit; ++i) {
    ++n_holes;
    ++n_ex;
    --anA;
    aZ -= target.hit[i].charge;
    ex_energy += target.hit[i].binding_energy;
    target4 -= target.hit[i].momentum;
  }

  // The projectile's.
  int anAb = projectile.a;
  int aZb = projectile.z;
  int n_exB = 0, n_chB = 0, n_holesB = 0;
  double ex_energyB = 0.0;
  LorentzVector projectile4;
  for (int i = 0; i < projectile.n_hit; ++i) {
    ++n_holesB;
    ++n_exB;
    --anAb;
    aZb -= projectile.hit[i].charge;
    ex_energyB += projectile.hit[i].binding_energy;
    projectile4 -= projectile.hit[i].momentum;
  }

  const bool exist_target_remnant =
      static_cast<double>(n_holes) < 0.3 * static_cast<double>(n_holes + anA);
  const bool exist_projectile_remnant =
      static_cast<double>(n_holesB) < 0.3 * static_cast<double>(n_holesB + anAb);

  const Vec3d bst = primary_projectile.boost_vector();
  const double R = target.radius;
  const double Rb = projectile.radius;

  for (int i = 0; i < n_tracks; ++i) {
    const CascadeTrack& t = tracks[i];
    const bool is_nucleon = (t.pdg == 2212 || t.pdg == 2112);
    if (!is_nucleon) {
      if (n_escaped >= escaped_capacity) { ref.capacity = true; return out; }
      escaped[n_escaped++] = t;
      continue;
    }

    const double tmag = t.momentum.mag();
    bool by_target = false;
    if (exist_target_remnant) {
      const double add = (t.momentum + target4).mag() - tmag - target4.mag();
      by_target = (-capture_threshold() * std::log(rng.uniform()) > add) &&
                  (g4gpu::mag(t.position) < R);
    }

    // The position four-vector boosted into the projectile's frame. Geant4 builds it
    // unconditionally, before either capture test, so it costs nothing to do the same.
    LorentzVector pos4(t.position, t.formation_time);
    pos4.boost(bst);

    bool by_projectile = false;
    if (exist_projectile_remnant) {
      const double add = (t.momentum + projectile4).mag() - tmag - projectile4.mag();
      by_projectile = (-capture_threshold() * std::log(rng.uniform()) > add) &&
                      (g4gpu::mag(pos4.v) < Rb);
    }

    if (by_target && by_projectile) {
      if (rng.uniform() < 0.5) { by_projectile = false; }
      else { by_target = false; }
    }

    if (by_target) {
      ++anA;
      ++n_ex;
      aZ += t.charge;
      n_ch += t.charge;
      target4 += t.momentum;
    } else if (by_projectile) {
      ++anAb;
      ++n_exB;
      aZb += t.charge;
      n_chB += t.charge;
      projectile4 += t.momentum;
    } else {
      if (n_escaped >= escaped_capacity) { ref.capacity = true; return out; }
      escaped[n_escaped++] = t;
    }
  }

  // The target residual. Two fix-ups, in this order: an untouched nucleus with no excitation
  // is put exactly at rest at its ground-state mass, and then a residual lighter than its own
  // ground state is raised to `fMass + exEnergy` using the accumulated binding energies.
  if (anA != 0) {
    const double fmass = deex::nuclear_mass(anA, aZ);
    if ((anA == target.a) && (ex_energy <= 0.0)) { target4.e = fmass; }
    const double remn = target4.mag();
    double ex = ex_energy;
    if (remn < fmass) {
      const double m = fmass + ex;
      target4.e = std::sqrt(g4gpu::mag2(target4.v) + m * m);
    } else {
      ex = remn - fmass;
    }
    if (ex < 0.0) { ex = 0.0; }
    out.target.fragment = deex::make_fragment(anA, aZ, target4);
    out.target.excitons.particles = n_ex - n_holes;
    out.target.excitons.charged = n_ch;
    out.target.excitons.holes = n_holes;
    out.target.exists = true;
  }

  // The projectile residual. Note that the `anAb == initial && exEnergyB <= 0` test restores
  // the PRIMARY's four-momentum rather than a rest mass - which is the right thing for a
  // projectile that was never touched - and that it sits OUTSIDE the `0 != anAb` block, so it
  // runs even when there is no projectile residual left.
  if ((anAb == projectile.a) && (ex_energyB <= 0.0)) { projectile4 = primary_projectile; }

  if (anAb != 0) {
    const double fmass = deex::nuclear_mass(anAb, aZb);
    const double remn = projectile4.mag();
    double ex = ex_energyB;
    if (remn < fmass) {
      const double m = fmass + ex;
      projectile4.e = std::sqrt(g4gpu::mag2(projectile4.v) + m * m);
    } else {
      ex = remn - fmass;
    }
    if (ex < 0.0) { ex = 0.0; }
    // findBoostToCM() is -p/E: the projectile residual is de-excited AT REST and its products
    // are boosted back by the caller.
    const Vec3d to_cm{-projectile4.v.x / projectile4.e, -projectile4.v.y / projectile4.e,
                      -projectile4.v.z / projectile4.e};
    out.projectile_boost_to_cm = to_cm;
    projectile4.boost(to_cm);
    out.projectile.fragment = deex::make_fragment(anAb, aZb, projectile4);
    out.projectile.excitons.particles = n_exB - n_holesB;
    out.projectile.excitons.charged = n_chB;
    out.projectile.excitons.holes = n_holesB;
    out.projectile.exists = true;
  }
  return out;
}

}  // namespace g4gpu::preco

#endif
