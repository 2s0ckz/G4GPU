// G4NucleiModel's cascade half: where a projectile enters the nucleus, what it meets there, and
// what one step of the intra-nuclear cascade does to it.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/G4NucleiModel.cc:
//   initializeCascad(G4InuclElementaryParticle*), initializeCascad(bullet, target, lists),
//   choosePointAlongTraj, generateNucleon, generateInteractionPartners, generateParticleFate
// and cascade/cascade/src/G4CascadParticle.cc (the five-argument constructor, which is
// `cp_fill` in workspace.cuh).
//
// P10's `nuclei_model.cuh` holds everything this needs and computes nothing random. This file
// is the part that draws.
//
// ---------------------------------------------------------------------------------------------
// Five things a reading of these four functions does not give you.
//
// **1. The projectile is told from its secondaries by a sentinel in a distance field.**
// `initializeCascad` gives the incident particle `current_path = large` (1000) where every
// secondary is created with 0, and `G4CascadParticle::young` - the veto that stops a freshly
// made secondary from interacting within a quarter of sqrt(10) - is
// `(current_path < 1000.) && (cpath < cut)`. So the beam particle is exempt because its path
// field holds a number no real path reaches, not because anything marks it. Two other things
// key off the same particle: `generation == 0` is `isProjectile`, and `forceFirst` - photons and
// muons must interact on their first step - is `isProjectile && (photon || muon)`.
//
// **2. `initializeCascad` starts the particle OUTSIDE the nucleus, in zone `number_of_zones`,
// unless it is at rest.** `getZone` returns `number_of_zones` for any radius at or beyond the
// surface, and the cascader's `stillInside` is `current_zone < number_of_zones`, so a particle
// that starts there is already "outside" and is carried in by the first `boundaryTransition`.
// The exception is capture at rest (`getKineticEnergy() < small`), which is put in the outermost
// real zone instead, because a particle with no momentum can never be transported inward.
//
// **3. The entry point is thrown as `sqrt(1-u)`, and the comment above it says the previous
// version did something else.** `costh = sqrt(1 - inuclRndm())` then
// `generateWithFixedTheta(-costh, nuclei_radius)`: a MINUS sign, so the particle starts on the
// hemisphere facing the beam, and the distribution is uniform in cos^2, which is uniform in
// impact parameter squared - i.e. uniform over the projected disc. Geant4's own FIXME records
// that the earlier code generated a random sin(theta) and used -cos(theta), and that switching
// to `generateWithRandomAngles` "changes result". It is one deviate, not two.
//
// **4. `choosePointAlongTraj` is for photons only, and it re-enters the nucleus along the whole
// chord.** A photon's mean free path is long compared with a nucleus, so starting it at the
// surface and asking it to interact there would bias every photonuclear event to the skin. The
// routine builds the chord from the entry point to the exit point, cuts it at every zone
// boundary, weights each segment by the integral of exp(-l/mfp) across it, and samples one
// point from that CDF. It is called from `initializeCascad` only when `forceFirst` is true.
// The weight uses the SUM of the proton and neutron inverse mean free paths in that zone, both
// computed against a target AT REST - `neutronEP`/`protonEP` are const members built with the
// one-argument constructor, whose four-momentum is (0,0,0,m).
//
// **5. `generateInteractionPartners` samples a nucleon for every species that could be met, and
// the list it builds is not the list of things that happen.** It throws one proton and one
// neutron (each with Fermi momentum), computes an interaction length for each, and keeps the
// ones that would occur before the zone boundary; then, for a pion, muon or photon, it throws up
// to three quasi-deuterons and picks at most one of them by cross section. The list is sorted by
// path and a dummy entry carrying the GEOMETRIC path to the boundary is appended last. So
// `thePartners.size() == 1` means "nothing interacts, move to the next zone", and any larger size
// means "try these in order". Every nucleon it throws costs three deviates and every
// quasi-deuteron six, whether or not the partner is used, so the draw count of one cascade step
// is a function of the nucleus's remaining census and not only of what happened.
//
// ---------------------------------------------------------------------------------------------
// **What `generateParticleFate` does with a collision it accepts, and the one it does not.**
// `passFermi` rejects a final state with a nucleon below the local Fermi momentum - Pauli
// blocking - and `continue`s to the next partner. `passTrailing` rejects an interaction too close
// to a previous one and is inert with the dumped `radiusTrailing` of 0 (see nuclei_model.cuh).
// A collision that produces NOTHING - `EPCoutput.numberOfOutgoingParticles() == 0`, which is
// `generateSCMfinalState` exhausting its ten attempts - does not `continue`: it `break`s out of
// the partner loop entirely and the particle is propagated to the boundary instead. Blocking is
// per-partner, kinematic failure is per-step.
#ifndef G4GPU_BERTINI_CASCADE_MODEL_CUH
#define G4GPU_BERTINI_CASCADE_MODEL_CUH

#include <cmath>

#include "physics/hadronic/bertini/ep_collider.cuh"
#include "physics/hadronic/bertini/nuclei_model.cuh"
#include "physics/hadronic/bertini/workspace.cuh"

namespace g4gpu::physics::hadronic::bert {

/// What a cascade step could not do. Every one is reported by name, never absorbed.
enum class FateRefusal : int {
  kNone = 0,
  kNegativePath,            ///< getPathToTheNextZone returned < -small: "something wrong"
  kEmptyPartnerList,        ///< generateParticleFate's own "got empty interaction-partners list"
  kTrajectoryDegenerate,    ///< choosePointAlongTraj: fewer than two zone crossings on the chord
  kTrajectoryNoWeight,      ///< choosePointAlongTraj: the CDF's last entry is not positive
  kTrajectorySampleOffEnd,  ///< choosePointAlongTraj: upper_bound ran off the end
  kCascadeOverflow,         ///< a workspace capacity; ws.overflow says which
  kColliderRefused,         ///< ep_collide reported a refusal; see ColliderOutput
  kFermiZoneOutOfSurface,   ///< passFermi asked for the Fermi momentum of the zone OUTSIDE the
                            ///< nucleus, which reads off the end of `fermi_momenta` in Geant4.
                            ///< Reachable only from a particle captured at rest that still has
                            ///< `current_zone == number_of_zones` - i.e. muon capture, P12's.
  kIonCoordinatesFailed,    ///< initializeCascad(nucleus): itry_max without a configuration
  kIonMomentaFailed         ///< initializeCascad(nucleus): itry_max without a momentum
};

// =============================================================================================
// CLHEP geometry that choosePointAlongTraj needs and core/vec3.cuh does not have
// =============================================================================================

/// CLHEP's clamp before the acos, on its own so that it can be asserted on its own.
///
/// `Hep3Vector::angle` divides a dot product by the square root of a product of magnitudes, and
/// for a nearly parallel pair that quotient can land an ulp above 1 - where `acos` returns NaN.
/// Through `choosePointAlongTraj`, the only caller, the arguments are an exactly normalised
/// position and either (0,0,1) or a normalised momentum, so the quotient is 1.0 exactly and the
/// clamp never fires: removing it passed all 538,874 comparisons. That makes it a question
/// answered by construction rather than a hole (docs/RISK.md V52), and this is the function
/// tests/test_bertini_cascade.cu asks directly.
__host__ __device__ inline double clhep_acos_clamped(double arg) {
  if (arg > 1.0) { arg = 1.0; }
  if (arg < -1.0) { arg = -1.0; }
  return std::acos(arg);
}

/// `Hep3Vector::angle(q)` - the angle between two vectors, with the cosine clamped into
/// [-1, 1] before the acos.
__host__ __device__ inline double clhep_angle(const Vec3d& a, const Vec3d& b) {
  const double ptot2 = g4gpu::mag2(a) * g4gpu::mag2(b);
  if (ptot2 <= 0.0) { return 0.0; }
  return clhep_acos_clamped(g4gpu::dot(a, b) / std::sqrt(ptot2));
}

/// `Hep3Vector::rotate(angle, axis)`, which is `HepRotation::rotate` applied to the vector:
/// Rodrigues' rotation about the NORMALISED axis, written as the 3x3 matrix CLHEP builds so
/// that the arithmetic is the same one. The `angle != 0` guard is CLHEP's - a zero angle leaves
/// the rotation the identity and the vector untouched, which is not the same expression as the
/// matrix evaluated at zero.
__host__ __device__ inline Vec3d clhep_rotate(const Vec3d& v, double angle, const Vec3d& axis) {
  if (angle == 0.0) { return v; }
  const double ll = g4gpu::mag(axis);
  if (ll == 0.0) { return v; }     // CLHEP prints "zero axis" and leaves the rotation alone
  const double sa = std::sin(angle);
  const double ca = std::cos(angle);
  const double dx = axis.x / ll, dy = axis.y / ll, dz = axis.z / ll;
  const double omca = 1.0 - ca;
  return Vec3d{
      (ca + omca * dx * dx) * v.x + (omca * dx * dy - sa * dz) * v.y +
          (omca * dx * dz + sa * dy) * v.z,
      (omca * dy * dx + sa * dz) * v.x + (ca + omca * dy * dy) * v.y +
          (omca * dy * dz - sa * dx) * v.z,
      (omca * dz * dx - sa * dy) * v.x + (omca * dz * dy + sa * dx) * v.y +
          (ca + omca * dz * dz) * v.z};
}

// =============================================================================================
// G4NucleiModel::generateNucleon and the entry point
// =============================================================================================

/// G4NucleiModel::generateNucleon(type, zone) - a nucleon with Fermi momentum, AS STORED.
///
/// The `G4InuclElementaryParticle(mom, type)` constructor runs `setMomentum`, so the four-vector
/// the caller gets back is the one that survives the round trip through a G4DynamicParticle and
/// not the one `generateNucleonMomentum` built. For a nucleon the two differ in the last ulp,
/// which is invisible - but this is the same store that moves a photon's energy in the ninth
/// digit (docs/RISK.md V125), and the rule is to put it where Geant4 puts it rather than where it
/// happens to matter.
template <typename Rng>
__host__ __device__ inline LV nm_generate_nucleon(const NucleiModel& m, int type, int zone,
                                                  Rng& rng) {
  return inucl_store_momentum(nm_generate_nucleon_momentum(m, type, zone, rng), type);
}

/// G4NucleiModel::choosePointAlongTraj - move an inbound particle to a point sampled along its
/// whole chord through the nucleus, weighted by interaction probability.
///
/// Called only for a photon or muon projectile (`forceFirst`). The particle is assumed to be ON
/// the surface, which is what `initializeCascad` just made it.
///
/// Two things worth keeping: the exit point is found by REFLECTING the entry point through the
/// chord (a rotation by `2*prang - pi` about `phat x rhat`), with a radial-incidence shortcut at
/// `prang < 1e-6` that sets it to `-pos` exactly; and the zone crossings are walked from the
/// OUTSIDE IN, `iz = zoneout - i`, so that the CDF accumulates in the direction of travel.
///
/// Geant4 has no guard on the three degenerate cases below - an empty crossing list, a zero
/// total weight (every zone transparent, so `wt` is 0/0), and a deviate that lands past the last
/// CDF entry. Each would be a silent NaN or an out-of-bounds read; each is a named refusal here
/// and the particle is left where it was.
template <typename Rng>
__host__ __device__ inline void nm_choose_point_along_traj(const NucleiModel& m,
                                                           const NucleiModelParams& p,
                                                           CascadeParticle& cp, Rng& rng,
                                                           FateRefusal& refusal) {
  refusal = FateRefusal::kNone;
  const Vec3d pos = cp.position;
  const Vec3d rhat = g4gpu::normalize(pos);

  Vec3d phat = g4gpu::normalize(cp.momentum.v);
  if (g4gpu::mag(cp.momentum.v) < kNmSmall) { phat = Vec3d{0.0, 0.0, 1.0}; }

  Vec3d posout = pos;
  const double prang = clhep_angle(rhat, Vec3d{-phat.x, -phat.y, -phat.z});
  if (prang < 1e-6) {
    posout = Vec3d{-pos.x, -pos.y, -pos.z};      // radial incidence
  } else {
    const double posrot = 2.0 * prang - 3.14159265358979323846;
    posout = clhep_rotate(posout, posrot, g4gpu::cross(phat, rhat));
  }

  const Vec3d posmid = (pos + posout) * 0.5;
  const double r2mid = g4gpu::mag2(posmid);
  const double lenmid = g4gpu::mag(posout - pos) * 0.5;

  const int zoneout = m.number_of_zones - 1;
  const int zonemid = nm_zone(m, g4gpu::mag(posmid));
  const int ncross = (m.number_of_zones - zonemid) * 2;
  if (ncross < 2) { refusal = FateRefusal::kTrajectoryDegenerate; return; }

  double wtlen[2 * kMaxZones];
  double len[2 * kMaxZones];
  for (int i = 0; i < ncross; ++i) { wtlen[i] = 0.0; len[i] = 0.0; }

  for (int i = 0; i < ncross / 2; ++i) {
    const int iz = zoneout - i;
    const double ds = std::sqrt(m.zone_radii[iz] * m.zone_radii[iz] - r2mid);
    len[i] = lenmid - ds;
    len[ncross - 1 - i] = lenmid + ds;
  }

  // `neutronEP` and `protonEP` are const members built by the one-argument constructor, i.e.
  // nucleons AT REST: (0, 0, 0, m). The kinetic energy the cross-section tables are then looked
  // up at is the projectile's own lab energy, not a Fermi-smeared one.
  const LV neutron_at_rest(Vec3d{0.0, 0.0, 0.0}, inucl_particle_mass(kNeutron));
  const LV proton_at_rest(Vec3d{0.0, 0.0, 0.0}, inucl_particle_mass(kProton));

  for (int i = 1; i < ncross; ++i) {
    // Outbound half and inbound half index the same zones in opposite order; `iz` is the zone
    // the segment [len[i-1], len[i]] lies in.
    const int iz = (i < ncross / 2) ? (zoneout - i + 1) : (zoneout - ncross + i + 1);
    bool ref_n = false, ref_p = false;
    const double invmfp =
        nm_inverse_mean_free_path(m, cp.type, cp.momentum, kNeutron, neutron_at_rest, iz, p,
                                  ref_n) +
        nm_inverse_mean_free_path(m, cp.type, cp.momentum, kProton, proton_at_rest, iz, p,
                                  ref_p);
    const double wt = (std::exp(-len[i - 1] * invmfp) - std::exp(-len[i] * invmfp)) / invmfp;
    wtlen[i] = wtlen[i - 1] + wt;
  }

  const double total = wtlen[ncross - 1];
  if (!(total > 0.0)) { refusal = FateRefusal::kTrajectoryNoWeight; return; }
  for (int i = 0; i < ncross; ++i) { wtlen[i] /= total; }

  const double rand = rng.uniform();
  int ir = 0;
  while (ir < ncross && !(wtlen[ir] > rand)) { ++ir; }   // std::upper_bound
  if (ir <= 0 || ir >= ncross) { refusal = FateRefusal::kTrajectorySampleOffEnd; return; }

  const double frac = (rand - wtlen[ir - 1]) / (wtlen[ir] - wtlen[ir - 1]);
  const double drand = (1.0 - frac) * len[ir - 1] + frac * len[ir];

  cp.position = pos + phat * drand;
  cp.current_zone = nm_zone(m, g4gpu::mag(cp.position));
}

/// G4NucleiModel::initializeCascad(G4InuclElementaryParticle*) - put the projectile on the
/// nuclear surface.
///
/// One deviate for the entry point, plus whatever `choosePointAlongTraj` costs for a photon or
/// muon. `large` (1000) in the path field is the sentinel that exempts this particle from the
/// young-secondary veto; see note 1 in the header.
template <typename Rng>
__host__ __device__ inline CascadeParticle nm_initialize_cascad(const NucleiModel& m,
                                                                const NucleiModelParams& p,
                                                                int type, const LV& mom,
                                                                Rng& rng,
                                                                FateRefusal& refusal) {
  refusal = FateRefusal::kNone;
  const double costh = std::sqrt(1.0 - rng.uniform());
  const Vec3d pos = inucl_with_fixed_theta(-costh, m.nuclei_radius, 0.0, rng).v;

  // `getKineticEnergy()` here is the STORED kinetic energy of the bullet - the interface built
  // it with fill(), so the round trip has already happened and `e - m` would be a different
  // number in the last ulp. It is compared against `small` = 1e-9 GeV = 1 eV.
  int zone = m.number_of_zones;
  if (inucl_stored_kinetic_energy(mom, type) < kNmSmall) { --zone; }

  CascadeParticle cp = cp_fill(type, mom, pos, zone, kNmLarge, 0);
  if (nm_force_first(cp.generation, cp.type)) {
    nm_choose_point_along_traj(m, p, cp, rng, refusal);
  }
  return cp;
}

// =============================================================================================
// G4NucleiModel::generateInteractionPartners
// =============================================================================================

/// Fills `ws.partners` with the candidate collisions for one step, sorted by path, with the
/// geometric path to the next zone boundary appended last.
///
/// `sortPartners` is `p2.second > p1.second`, i.e. ascending path, and `std::sort` on a range of
/// at most three elements is an insertion sort in every implementation this port is built with;
/// an insertion sort is written out here because it is the one that agrees on ties, and two
/// independently sampled path lengths do tie when both come back as `large`.
template <typename Rng>
__host__ __device__ inline void nm_generate_interaction_partners(NucleiModel& m,
                                                                 const NucleiModelParams& p,
                                                                 CascadeParticle& cp,
                                                                 BertiniWorkspace& ws, Rng& rng,
                                                                 FateRefusal& refusal) {
  refusal = FateRefusal::kNone;
  ws.n_partners = 0;

  const int ptype = cp.type;
  int zone = cp.current_zone;

  double r_in, r_out;
  if (zone == m.number_of_zones) {
    r_in = m.nuclei_radius;
    r_out = 0.0;
  } else if (zone == 0) {
    r_in = 0.0;
    r_out = m.zone_radii[0];
  } else {
    r_in = m.zone_radii[zone - 1];
    r_out = m.zone_radii[zone];
  }

  bool moving_in = cp.moving_in;
  const double path = cp_path_to_next_zone(cp.position, cp.momentum, cp.current_zone, r_in,
                                           r_out, moving_in);
  cp.moving_in = moving_in;    // getPathToTheNextZone SETS movingIn as a side effect

  if (path < -kNmSmall) { refusal = FateRefusal::kNegativePath; return; }

  if (std::fabs(path) < kNmSmall) {         // not moving, or exactly at a boundary
    if (g4gpu::mag(cp.momentum.v) > kNmSmall) {
      // A zero path with real momentum: emit ONLY the dummy terminator, with path zero, so
      // that generateParticleFate's `npart == 1` arm moves the particle nowhere and lets
      // boundaryTransition act. The dummy's type is 0 (G4InuclElementaryParticle's default).
      ws.partners[0] = InteractionPartner{0, LV(), 0.0};
      ws.n_partners = 1;
      return;
    }
    // Captured at rest outside the nucleus: place it in the outermost real zone. Note this
    // changes the LOCAL `zone` used for generateNucleon below and NOT `cp.current_zone`, which
    // is what inverseMeanFreePath's default argument reads - so the two can disagree here.
    if (zone >= m.number_of_zones) { zone = m.number_of_zones - 1; }
  }

  double invmfp = 0.0;
  double spath = 0.0;
  for (int ip = 1; ip < 3; ++ip) {
    if (ip == kProton && m.proton_number_current < 1) { continue; }
    if (ip == kNeutron && m.neutron_number_current < 1) { continue; }
    if (ip == kNeutron && ptype == kMuonMinus) { continue; }   // mu-/n forbidden

    const LV target = nm_generate_nucleon(m, ip, zone, rng);
    bool ref = false;
    invmfp = nm_inverse_mean_free_path(m, ptype, cp.momentum, ip, target, cp.current_zone, p,
                                       ref);
    spath = nm_generate_interaction_length(path, invmfp, nm_force_first(cp.generation, ptype),
                                           cp.current_path, rng);
    if (path < kNmSmall || spath < path) {
      if (ws.n_partners >= kMaxPartners) {
        ws.overflow = CascadeOverflow::kPartners;
        refusal = FateRefusal::kCascadeOverflow;
        return;
      }
      ws.partners[ws.n_partners++] = InteractionPartner{ip, target, spath};
    }
  }

  if (nm_use_quasideuteron(ptype)) {
    ws.n_qdeutrons = 0;
    double tot_invmfp = 0.0;

    // pp interacts with pi-, mu- or neutrals; np with any pion or photon; nn with pi+ or
    // neutrals. The three tests are on the CURRENT census, so a nucleus the cascade has eaten
    // into offers fewer dibaryons.
    if (m.proton_number_current >= 2 && ptype != kPionPlus) {
      int dtype = 0;
      const LV d = inucl_store_momentum(
          nm_generate_quasideuteron(m, kProton, kProton, zone, dtype, rng), kDiproton);
      bool ref = false;
      invmfp = nm_inverse_mean_free_path(m, ptype, cp.momentum, dtype, d, cp.current_zone, p,
                                         ref);
      tot_invmfp += invmfp;
      ws.acsecs[ws.n_qdeutrons] = invmfp;
      ws.qdeutrons[ws.n_qdeutrons] = d;
      ws.qdeutron_types[ws.n_qdeutrons] = dtype;
      ++ws.n_qdeutrons;
    }
    if (m.proton_number_current >= 1 && m.neutron_number_current >= 1) {
      int dtype = 0;
      const LV d = inucl_store_momentum(
          nm_generate_quasideuteron(m, kProton, kNeutron, zone, dtype, rng), kUnboundPN);
      bool ref = false;
      invmfp = nm_inverse_mean_free_path(m, ptype, cp.momentum, dtype, d, cp.current_zone, p,
                                         ref);
      tot_invmfp += invmfp;
      ws.acsecs[ws.n_qdeutrons] = invmfp;
      ws.qdeutrons[ws.n_qdeutrons] = d;
      ws.qdeutron_types[ws.n_qdeutrons] = dtype;
      ++ws.n_qdeutrons;
    }
    if (m.neutron_number_current >= 2 && ptype != kPionMinus && ptype != kMuonMinus) {
      int dtype = 0;
      const LV d = inucl_store_momentum(
          nm_generate_quasideuteron(m, kNeutron, kNeutron, zone, dtype, rng), kDineutron);
      bool ref = false;
      invmfp = nm_inverse_mean_free_path(m, ptype, cp.momentum, dtype, d, cp.current_zone, p,
                                         ref);
      tot_invmfp += invmfp;
      ws.acsecs[ws.n_qdeutrons] = invmfp;
      ws.qdeutrons[ws.n_qdeutrons] = d;
      ws.qdeutron_types[ws.n_qdeutrons] = dtype;
      ++ws.n_qdeutrons;
    }

    if (tot_invmfp > kNmSmall) {
      const double apath = nm_generate_interaction_length(
          path, tot_invmfp, nm_force_first(cp.generation, ptype), cp.current_path, rng);
      if (path < kNmSmall || apath < path) {
        const double sl = rng.uniform() * tot_invmfp;
        double as = 0.0;
        for (int i = 0; i < ws.n_qdeutrons; ++i) {
          as += ws.acsecs[i];
          if (sl < as) {
            if (ws.n_partners >= kMaxPartners) {
              ws.overflow = CascadeOverflow::kPartners;
              refusal = FateRefusal::kCascadeOverflow;
              return;
            }
            ws.partners[ws.n_partners++] =
                InteractionPartner{ws.qdeutron_types[i], ws.qdeutrons[i], apath};
            break;
          }
        }
      }
    }
  }

  if (ws.n_partners > 1) {     // std::sort(thePartners, sortPartners): ascending path
    for (int i = 1; i < ws.n_partners; ++i) {
      const InteractionPartner key = ws.partners[i];
      int j = i - 1;
      while (j >= 0 && key.path < ws.partners[j].path) {
        ws.partners[j + 1] = ws.partners[j];
        --j;
      }
      ws.partners[j + 1] = key;
    }
  }

  // The total-path placeholder, appended AFTER the sort so that it is always last regardless of
  // how the real partners compare against the geometric path.
  if (ws.n_partners >= kMaxPartners) {
    ws.overflow = CascadeOverflow::kPartners;
    refusal = FateRefusal::kCascadeOverflow;
    return;
  }
  ws.partners[ws.n_partners++] = InteractionPartner{0, LV(), path};
}

// =============================================================================================
// G4NucleiModel::generateParticleFate
// =============================================================================================

/// G4NucleiModel::passFermi over a collider's output, without copying the moduli into a local
/// array. `getMomModule()` is `|p|` of the STORED four-vector, which is what the collider left
/// in `ep_out.momenta` after its two `setMomentum` round trips.
__host__ __device__ inline bool fate_pass_fermi(const NucleiModel& m,
                                                const ColliderOutput& ep_out, int zone) {
  for (int i = 0; i < ep_out.n; ++i) {
    if (!inucl_is_nucleon(ep_out.kinds[i])) { continue; }
    if (g4gpu::mag(ep_out.momenta[i].v) < m.fermi_momenta[ep_out.kinds[i] - 1][zone]) {
      return false;
    }
  }
  return true;
}

/// One step of the cascade: propagate, or collide.
///
/// The outgoing particles go into `ws.new_cascade` / `ws.n_new_cascade`, which is Geant4's
/// `new_cascad_particles` buffer. Exactly one of two things is written there: ONE particle (the
/// same one, moved and boundary-transitioned) or the products of ONE collision. The caller tells
/// the two apart by the count, which is what G4IntraNucleiCascader does.
///
/// `cp` is modified in place, as Geant4 modifies its `cparticle` argument: the position is
/// rewound to `old_position` before each partner is tried, the zone is the one the step STARTED
/// in (`zone` is read once, before the loop, and every product is created in it), and the
/// momentum is only changed on the no-interaction path.
template <typename Rng>
__host__ __device__ inline void nm_generate_particle_fate(NucleiModel& m,
                                                          const NucleiModelParams& p,
                                                          const CascadeParams& par,
                                                          CascadeParticle& cp,
                                                          ColliderOutput& ep_out,
                                                          BertiniWorkspace& ws, Rng& rng,
                                                          FateRefusal& refusal) {
  refusal = FateRefusal::kNone;
  ws.n_new_cascade = 0;

  nm_generate_interaction_partners(m, p, cp, ws, rng, refusal);
  if (refusal != FateRefusal::kNone) { return; }
  if (ws.n_partners == 0) { refusal = FateRefusal::kEmptyPartnerList; return; }

  const int npart = ws.n_partners;

  if (npart == 1) {           // nothing interacts: move to the next zone
    cp.position = cp_propagate(cp.position, cp.momentum, ws.partners[0].path);
    cp.current_path += ws.partners[0].path;
    bool in_zone_zero = false;
    nm_boundary_transition(m, cp.type, cp.position, cp.momentum, cp.current_zone, cp.moving_in,
                           cp.reflection_counter, cp.reflected, in_zone_zero);
    ws.new_cascade[ws.n_new_cascade++] = cp;
    // "A.R. 19-Jun-2013: Fixed rare cases of non-reproducibility" - the pair has to be cleared
    // on this path too, or the cascader turns the PREVIOUS step's nucleons into holes again.
    m.current_nucl1 = 0;
    m.current_nucl2 = 0;
    return;
  }

  const Vec3d old_position = cp.position;
  bool no_interaction = true;
  const int zone = cp.current_zone;

  for (int i = 0; i < npart - 1; ++i) {
    if (i > 0) { cp.position = old_position; }

    const int target_type = ws.partners[i].type;
    const LV target_mom = ws.partners[i].momentum;

    // setNucleusState(A_current, Z_current) then collide(). The (A, Z) is the CURRENT census,
    // not the original nucleus: only generateSCMpionNAbsorption reads it, and what it needs is
    // the mass of what is left after this nucleon is removed.
    const int mass_number_current = m.proton_number_current + m.neutron_number_current;
    ep_collide_in_nucleus(cp.type, cp.momentum, target_type, target_mom, mass_number_current,
                          m.proton_number_current, true, par, ep_out, ws, rng);

    if (ep_out.n == 0) {
      // "If collision failed, exit loop over partners" - a break, not a continue.
      //
      // An empty final state is a normal outcome of Geant4's own collider and most of the
      // verdicts that produce one are Geant4's, not this port's: ten failed kinematic attempts
      // (docs/RISK.md V121), a missing channel table, an illegal dibaryon partner. Those break
      // the loop and the particle is propagated instead, which is what Geant4 does. Only the
      // four that mean "this port declines" are raised to the caller - see
      // `collider_refusal_is_port_limit`. Treating the whole enum as a refusal makes a photon at
      // rest in carbon abort the cascade, which is where this distinction was found.
      if (collider_refusal_is_port_limit(ep_out.refusal)) {
        refusal = FateRefusal::kColliderRefused;
      }
      break;
    }

    // passFermi reads `fermi_momenta[type-1][zone]`, and `zone` here is the particle's own
    // current zone with no clamp in Geant4. A particle still marked as outside the surface
    // therefore reads one past the end of the table; refused by name rather than reproduced,
    // because what it reads is the allocator's, not the model's. See the enum.
    if (zone < 0 || zone >= m.number_of_zones) {
      refusal = FateRefusal::kFermiZoneOutOfSurface;
      return;
    }
    if (!fate_pass_fermi(m, ep_out, zone)) { continue; }

    cp.position = cp_propagate(cp.position, cp.momentum, ws.partners[i].path);
    if (!nm_pass_trailing(ws.collision_points, ws.n_collision_points, cp.position,
                          p.r_nucleon)) {
      continue;
    }
    if (!ws_push_collision_point(ws, cp.position)) {
      refusal = FateRefusal::kCascadeOverflow;
      return;
    }

    // std::sort(outgoing_particles, G4ParticleLargerBeta()) - descending beta, i.e. |p|/E of
    // the STORED four-vector. Not the same order as the descending-Ekin sort ep_collide already
    // applied: a pion and a nucleon of equal kinetic energy have very different betas, and this
    // is the order the cascade stack is then filled in, so it decides which product is stepped
    // first. Insertion sort, at most nine elements.
    for (int a = 1; a < ep_out.n; ++a) {
      const LV km = ep_out.momenta[a];
      const int kk = ep_out.kinds[a];
      const double kb = g4gpu::mag(km.v) / km.e;
      int b = a - 1;
      while (b >= 0 && (g4gpu::mag(ep_out.momenta[b].v) / ep_out.momenta[b].e) < kb) {
        ep_out.momenta[b + 1] = ep_out.momenta[b];
        ep_out.kinds[b + 1] = ep_out.kinds[b];
        --b;
      }
      ep_out.momenta[b + 1] = km;
      ep_out.kinds[b + 1] = kk;
    }

    const int next_gen = cp.generation + 1;
    for (int ip = 0; ip < ep_out.n; ++ip) {
      ws.new_cascade[ws.n_new_cascade++] =
          cp_fill(ep_out.kinds[ip], ep_out.momenta[ip], cp.position, zone, 0.0, next_gen);
    }

    no_interaction = false;
    m.current_nucl1 = 0;
    m.current_nucl2 = 0;

    // Which nucleons were consumed. A dibaryon's code carries both: 111 -> (1,1),
    // 112 -> (1,2), 122 -> (2,2), decoded by the same arithmetic Geant4 uses.
    if (inucl_is_nucleon(target_type)) {
      m.current_nucl1 = target_type;
    } else {
      m.current_nucl1 = (target_type - 100) / 10;
      m.current_nucl2 = target_type - 100 - 10 * m.current_nucl1;
    }

    // Note the ELSE: anything that is not a proton decrements the NEUTRON count, including the
    // current_nucl1 == 0 that a non-nucleon, non-dibaryon target would leave. It cannot happen
    // - the partner list holds only nucleons and dibaryons - and it is written as Geant4 writes
    // it rather than guarded, because a guard would be a different model.
    if (m.current_nucl1 == 1) { --m.proton_number_current; }
    else { --m.neutron_number_current; }

    if (m.current_nucl2 == 1) { --m.proton_number_current; }
    else if (m.current_nucl2 == 2) { --m.neutron_number_current; }

    break;
  }

  if (no_interaction) {
    cp.position = old_position;
    cp.position = cp_propagate(cp.position, cp.momentum, ws.partners[npart - 1].path);
    cp.current_path += ws.partners[npart - 1].path;
    bool in_zone_zero = false;
    nm_boundary_transition(m, cp.type, cp.position, cp.momentum, cp.current_zone, cp.moving_in,
                           cp.reflection_counter, cp.reflected, in_zone_zero);
    ws.new_cascade[ws.n_new_cascade++] = cp;
  }
}

// =============================================================================================
// G4NucleiModel::initializeCascad(bullet, target, lists) - a NUCLEUS projectile
// =============================================================================================

/// The ion-on-ion entry: break the projectile into nucleons, place them, and decide which of
/// them enter the target as cascade particles and which are released directly.
///
/// **QBBC never reaches this.** `G4HadronInelasticQBBC` registers Bertini on p, n, pi+ and pi-,
/// and `G4HadronicBuilder::BuildFTFP_BERT` on kaons and hyperons; ions go to
/// `G4BinaryLightIonReaction` and FTFP. It is transcribed because `G4CascadeInterface::
/// createBullet` builds a `G4InuclNuclei` for any projectile with `GetAtomicMass() > 1`, so a
/// physics list that hands Bertini an ion gets here, and because the brief names it.
///
/// Four things in it that a rewrite would silently change:
///
/// **The two nucleons of a deuteron bullet come out with the SAME momentum vector.** The code is
///     momentums.push_back(mom);
///     mom.setVect(-mom.vect());
///     momentums.push_back(-mom);
/// and unary minus on a four-vector negates all four components, so the second entry is
/// `(+v, -E)` - the three-momentum is back to where it started and the ENERGY is negative. It
/// does not produce a negative-energy particle, because `G4InuclElementaryParticle(mom, knd)` is
/// built with a NUCLEON type against a four-vector carrying the DEUTERON mass: the
/// `|getMass() - mom.m()| <= 1e-5` test fails, `setMomentum` takes its `SetMomentum(vect)` arm,
/// and the energy is discarded and rebuilt from the nucleon mass. What survives is the
/// three-momentum, and both nucleons have the same one.
///
/// **Every momentum four-vector is built with the BULLET's mass**, `generateWithRandomAngles(p,
/// massb)`, and then read back as a nucleon - so the same discard applies to all of them, and
/// the energies computed here never leave the function. Only the directions and moduli matter.
///
/// **A configuration accepted on the hundredth trial is followed by a bogus one.** The inner
/// samplers are `while (itry1 < itry_max) { itry1++; ... break; }` followed by
/// `if (itry1 == itry_max) { push (10000,10000,10000); break; }`, and a success on the last
/// trial leaves `itry1 == itry_max`, so the good coordinate AND the sentinel are both pushed.
/// Same shape as docs/RISK.md V121, and as unreachable: the acceptance probability per trial is
/// of order 0.3, so this needs 0.7^100 ~ 3e-16. Reproduced; the OUT-OF-BOUNDS read that follows
/// it when the break happens early is refused by name instead, because what Geant4 reads there
/// is not defined by the source.
///
/// **The bullet-frame boost uses a 1 keV proton as its reference.** `G4InuclElementaryParticle
/// dummy(small_ekin, 1)` is a proton along +z with 1e-6 GeV of kinetic energy, and it exists
/// only to give `G4LorentzConvertor` a bullet so that `toTheTargetRestFrame` can be asked for
/// the boost out of the real bullet's rest frame.
template <typename Rng>
__host__ __device__ inline void nm_initialize_cascad_nucleus(
    const NucleiModel& m, int ab, int zb, const LV& bullet_mom, double bullet_mass,
    double bullet_stored_ekin, int at, int zt, const LV& target_mom, BertiniWorkspace& ws,
    int* out_cascade_n, int* out_released_types, LV* out_released, int* out_released_n,
    int released_cap, Rng& rng, FateRefusal& refusal) {
  refusal = FateRefusal::kNone;
  *out_cascade_n = 0;
  *out_released_n = 0;
  ws.n_cascade = 0;

  constexpr double kMaxAForCascad = 5.0;
  constexpr double kEkinCut = 2.0;
  constexpr double kIonSmallEkin = 1.0e-6;
  constexpr double kRLarge2For3 = 62.0;
  constexpr double kR0ForAeq3 = 3.92;
  constexpr double kS3Max = 6.5;
  constexpr double kRLarge2For4 = 69.14;
  constexpr double kR0ForAeq4 = 4.16;
  constexpr double kS4Max = 7.0;
  constexpr int kIonItryMax = 100;

  if (!(double(ab) < kMaxAForCascad)) { return; }   // compound nucleus; both lists stay empty
  if (ab > BertiniWorkspace::kMaxIonBulletA) { return; }

  const double benb = inucl_binding_energy(ab, zb) * 0.001 / double(ab);
  const double bent = inucl_binding_energy(at, zt) * 0.001 / double(at);
  const double ben = (benb < bent) ? bent : benb;
  if (!(bullet_stored_ekin / double(ab) > kEkinCut * ben)) { return; }

  int itryg = 0;
  while (*out_cascade_n == 0 && itryg < kIonItryMax) {
    ++itryg;
    *out_released_n = 0;
    ws.n_ion_coordinates = 0;
    ws.n_ion_momenta = 0;

    if (ab < 3) {                                   // deuteron, simplest case
      const double r = 2.214 - 3.4208 * std::log(1.0 - 0.981 * rng.uniform());
      const Vec3d coord1 = inucl_with_random_angles(r, 0.0, rng).v;
      ws.ion_coordinates[0] = coord1;
      ws.ion_coordinates[1] = Vec3d{-coord1.x, -coord1.y, -coord1.z};
      ws.n_ion_coordinates = 2;

      double pp = 0.0;
      bool bad = true;
      int itry = 0;
      while (bad && itry < kIonItryMax) {
        ++itry;
        pp = 456.0 * rng.uniform();
        if (pp * pp / (pp * pp + 2079.36) / (pp * pp + 2079.36) > 1.2023e-4 * rng.uniform() &&
            pp * r > 312.0) {
          bad = false;
        }
      }
      pp = 0.0005 * pp;
      const LV mom = inucl_with_random_angles(pp, bullet_mass, rng);
      ws.ion_momenta[0] = mom;
      // `mom.setVect(-v)` then `-mom`: the three-vector comes back to +v and the energy is
      // negated. Both are written out because the second is what the source produces.
      ws.ion_momenta[1] = LV(mom.v, -mom.e);
      ws.n_ion_momenta = 2;
    } else {
      bool badco = true;
      int itry = 0;
      Vec3d coord1{0.0, 0.0, 0.0};

      if (ab == 3) {
        while (badco && itry < kIonItryMax) {
          if (itry > 0) { ws.n_ion_coordinates = 0; }
          ++itry;
          bool broke_early = false;
          for (int i = 0; i < 2; ++i) {
            int itry1 = 0;
            const double fmax = std::exp(-0.5) / std::sqrt(0.5);
            while (itry1 < kIonItryMax) {
              ++itry1;
              double ss = -std::log(rng.uniform());
              const double u = fmax * rng.uniform();
              const double rho = std::sqrt(ss) * std::exp(-ss);
              if (rho > u && ss < kS3Max) {
                ss = kR0ForAeq3 * std::sqrt(ss);
                coord1 = inucl_with_random_angles(ss, 0.0, rng).v;
                ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;
                break;
              }
            }
            if (itry1 == kIonItryMax) {     // fires on a hundredth-trial SUCCESS too
              coord1 = Vec3d{10000.0, 10000.0, 10000.0};
              if (ws.n_ion_coordinates < BertiniWorkspace::kMaxIonBulletA) {
                ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;
              }
              broke_early = true;
              break;
            }
          }
          if (broke_early && ws.n_ion_coordinates < 2) {
            refusal = FateRefusal::kIonCoordinatesFailed;   // Geant4 reads coordinates[1] here
            return;
          }
          coord1 = Vec3d{0.0, 0.0, 0.0} - ws.ion_coordinates[0] - ws.ion_coordinates[1];
          ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;

          bool large_dist = false;
          for (int i = 0; i < 2 && !large_dist; ++i) {
            for (int j = i + 1; j < 3; ++j) {
              const Vec3d d = ws.ion_coordinates[i] - ws.ion_coordinates[j];
              if (g4gpu::mag2(d) > kRLarge2For3) { large_dist = true; break; }
            }
          }
          if (!large_dist) { badco = false; }
        }
      } else {                                        // a >= 4
        const double b = 3.0 / (double(ab) - 2.0);
        const double b1 = 1.0 - b / 2.0;
        double u = b1 + std::sqrt(b1 * b1 + b);
        const double fmax = (1.0 + u / b) * u * std::exp(-u);

        while (badco && itry < kIonItryMax) {
          if (itry > 0) { ws.n_ion_coordinates = 0; }
          ++itry;
          bool broke_early = false;
          for (int i = 0; i < ab - 1; ++i) {
            int itry1 = 0;
            while (itry1 < kIonItryMax) {
              ++itry1;
              double ss = -std::log(rng.uniform());
              u = fmax * rng.uniform();
              if (std::sqrt(ss) * std::exp(-ss) * (1.0 + ss / b) > u && ss < kS4Max) {
                ss = kR0ForAeq4 * std::sqrt(ss);
                coord1 = inucl_with_random_angles(ss, 0.0, rng).v;
                ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;
                break;
              }
            }
            if (itry1 == kIonItryMax) {
              coord1 = Vec3d{10000.0, 10000.0, 10000.0};
              if (ws.n_ion_coordinates < BertiniWorkspace::kMaxIonBulletA) {
                ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;
              }
              broke_early = true;
              break;
            }
          }
          if (broke_early && ws.n_ion_coordinates < ab - 1) {
            refusal = FateRefusal::kIonCoordinatesFailed;
            return;
          }
          coord1 = Vec3d{0.0, 0.0, 0.0};
          for (int j = 0; j < ab - 1; ++j) { coord1 = coord1 - ws.ion_coordinates[j]; }
          ws.ion_coordinates[ws.n_ion_coordinates++] = coord1;

          bool large_dist = false;
          for (int i = 0; i < ab - 1 && !large_dist; ++i) {
            for (int j = i + 1; j < ab; ++j) {
              const Vec3d d = ws.ion_coordinates[i] - ws.ion_coordinates[j];
              if (g4gpu::mag2(d) > kRLarge2For4) { large_dist = true; break; }
            }
          }
          if (!large_dist) { badco = false; }
        }
      }

      if (badco) { refusal = FateRefusal::kIonCoordinatesFailed; return; }

      LV mom;
      for (int i = 0; i < ab - 1; ++i) {
        int itry2 = 0;
        bool got = false;
        while (itry2 < kIonItryMax) {
          ++itry2;
          const double uu = -std::log(0.879853 - 0.8798502 * rng.uniform());
          const double x = uu * std::exp(-uu);
          if (x > rng.uniform()) {
            const double pmod = std::sqrt(0.01953 * uu);
            mom = inucl_with_random_angles(pmod, bullet_mass, rng);
            ws.ion_momenta[ws.n_ion_momenta++] = mom;
            got = true;
            break;
          }
        }
        if (!got) { refusal = FateRefusal::kIonMomentaFailed; return; }
      }
      // The last nucleon carries whatever is left of the TOTAL energy of bullet plus target -
      // which is not a nucleon four-momentum at all, and is discarded by the mass mismatch on
      // the way into G4InuclElementaryParticle. Only its three-vector survives.
      mom = LV(Vec3d{0.0, 0.0, 0.0}, bullet_mom.e + target_mom.e);
      for (int j = 0; j < ab - 1; ++j) { mom -= ws.ion_momenta[j]; }
      ws.ion_momenta[ws.n_ion_momenta++] = mom;
    }

    // Coordinates and momenta are in the bullet's rest frame; put the cluster on an impact
    // point and boost the nucleons to the lab.
    double rb = 0.0;
    for (int i = 0; i < ws.n_ion_coordinates; ++i) {
      const double rp = g4gpu::mag2(ws.ion_coordinates[i]);
      if (rp > rb) { rb = rp; }
    }

    const double s1 = std::sqrt(rng.uniform());
    const double phi = inucl_random_phi(rng);
    const double rz = (m.nuclei_radius + rb) * s1;
    const Vec3d global_pos{rz * std::cos(phi), rz * std::sin(phi),
                           -(m.nuclei_radius + rb) * std::sqrt(1.0 - s1 * s1)};
    for (int i = 0; i < ws.n_ion_coordinates; ++i) {
      ws.ion_coordinates[i] = ws.ion_coordinates[i] + global_pos;
    }

    // `rb` is a squared radius used as a radius - `rp` is `mag2()` and the largest is kept
    // without a square root. Written as Geant4 writes it; it makes the impact disc larger than
    // the cluster actually is, which is conservative and not a correction this port may make.

    LorentzConvertor to_bullet_rest;
    to_bullet_rest.bullet =
        LV(Vec3d{0.0, 0.0, std::sqrt(kIonSmallEkin * (kIonSmallEkin +
                                                      2.0 * inucl_particle_mass(kProton)))},
           kIonSmallEkin + inucl_particle_mass(kProton));
    to_bullet_rest.target = bullet_mom;
    lc_to_the_target_rest_frame(to_bullet_rest);

    for (int ip = 0; ip < ab; ++ip) {
      const int knd = (ip < zb) ? kProton : kNeutron;
      const LV raw = inucl_store_momentum(ws.ion_momenta[ip], knd);
      const LV mom = inucl_store_momentum(lc_back_to_the_lab(to_bullet_rest, raw), knd);

      const double pmod = g4gpu::mag(mom.v);
      const double t0 = -g4gpu::dot(mom.v, ws.ion_coordinates[ip]) / pmod;
      const double det = t0 * t0 + m.nuclei_radius * m.nuclei_radius -
                         g4gpu::mag2(ws.ion_coordinates[ip]);
      double tr = -1.0;
      if (det > 0.0) {
        const double t1 = t0 + std::sqrt(det);
        const double t2 = t0 - std::sqrt(det);
        if (std::fabs(t1) <= std::fabs(t2)) {
          if (t1 > 0.0 && ws.ion_coordinates[ip].z + mom.v.z * t1 / pmod <= 0.0) { tr = t1; }
          if (tr < 0.0 && t2 > 0.0 &&
              ws.ion_coordinates[ip].z + mom.v.z * t2 / pmod <= 0.0) { tr = t2; }
        } else {
          if (t2 > 0.0 && ws.ion_coordinates[ip].z + mom.v.z * t2 / pmod <= 0.0) { tr = t2; }
          if (tr < 0.0 && t1 > 0.0 &&
              ws.ion_coordinates[ip].z + mom.v.z * t1 / pmod <= 0.0) { tr = t1; }
        }
      }

      if (tr >= 0.0) {            // enters the target: a cascade particle on the surface
        ws.ion_coordinates[ip] = ws.ion_coordinates[ip] + mom.v * (tr / pmod);
        if (!ws_push_cascade(ws, cp_fill(knd, mom, ws.ion_coordinates[ip], m.number_of_zones,
                                         kNmLarge, 0))) {
          refusal = FateRefusal::kCascadeOverflow;
          return;
        }
        ++(*out_cascade_n);
      } else {                    // misses: released straight to the output list
        if (*out_released_n >= released_cap) {
          ws.overflow = CascadeOverflow::kOutgoingParticles;
          refusal = FateRefusal::kCascadeOverflow;
          return;
        }
        out_released_types[*out_released_n] = knd;
        out_released[*out_released_n] = mom;
        ++(*out_released_n);
      }
    }
  }

  if (*out_cascade_n == 0) { *out_released_n = 0; }   // "can not generate proper distribution"
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_CASCADE_MODEL_CUH
