// Bertini's OWN de-excitation: the chain that runs when the cascade's residual is not handed to
// PreCompound.
//
// Transcribed from Geant4 11.1.1, cascade/cascade/src/:
//   G4CascadeDeexciteBase::getTargetData / makeFragment / explosion / validateOutput
//   G4CascadeDeexcitation::deExcite
//   G4BigBanger::deExcite / generateBangInSCM / generateMomentumModules / xProbability /
//     maxProbability / generateX
//   G4NonEquilibriumEvaporator::deExcite / getMatrixElement / getE0 / getParLev
//   G4EquilibriumEvaporator::deExcite / explosion / goodRemnant / getQF / getAF /
//     getPARLEVDEN / getE0
//   G4Fissioner::deExcite / getC2 / getZopt / potentialMinimization
//   G4FissionStore::addConfig / generateConfiguration
//
// ---------------------------------------------------------------------------------------------
// **Which de-excitation an event gets is decided by the physics-list builder, not by a
// parameter.** `G4CascadeParameters::usePreCompound()` is false, so `G4CascadeInterface`'s
// constructor calls `useCascadeDeexcitation()` - and then `G4HadronInelasticQBBC` calls
// `usePreCompoundDeexcitation()` on the p/n instance and on the pi+/pi- instance, overriding it.
// So in QBBC this file runs for KAONS AND HYPERONS ONLY, the instance
// `G4HadronicBuilder::BuildFTFP_BERT` makes, and nucleons and pions go to P6. docs/RISK.md V118.
// Both arms are kept and the choice is a parameter of `bert::deexcite`, because the flag is an
// environment variable and a run that sets it has to be reproducible.
//
// ---------------------------------------------------------------------------------------------
// **`explosion()` is two different functions with the same name**, and which one a fragment
// meets depends on where in the chain it is.
//
//   G4CascadeDeexciteBase::explosion(A, Z, E)   -> (A <= 20 || Z == 0) && E >= 3*BE(A,Z)
//   G4EquilibriumEvaporator::explosion(A, Z, E) -> !(A >= 12 && Z >= 0 && Z < 3*(A-Z))
//                                                  && E >= 3*BE(A,Z)
//
// `G4CascadeDeexcitation::deExcite` asks the BASE class's, because it derives from
// G4CascadeDeexciteBase; the equilibrium evaporator hides it with its own and asks that one
// twice, once on entry and once per iteration. They disagree in both directions: a fragment with
// A = 30 and Z = 0 explodes under the first and not the second, and one with A = 8 and Z = 6
// explodes under the second and not the first. The comment beside the second says why - "more
// agitated" - and both are transcribed under their own names.
//
// ---------------------------------------------------------------------------------------------
// **The equilibrium evaporator calls ITSELF**, once per fission fragment, and that is the only
// recursion in the whole Bertini tree. It is bounded by the fission threshold - `A >= 100` - so a
// uranium residual splits into two ~120 halves, each of which can split once more into ~60, and
// 60 cannot split. Three levels, not more. A device kernel does not recurse, so the fragments go
// on `BertiniWorkspace::deex_stack` and the loop drains it; the order is the same, because
// Geant4 evaporates fragment 0 to completion before touching fragment 1 and a stack drained from
// the top does the same when the second is pushed first.
//
// ---------------------------------------------------------------------------------------------
// Four things a reading does not give you.
//
// **1. `xProbability`'s guard is vacuous.** `if (x < 1.0 || x > 0.0)` is true for every real x -
// it would take `x >= 1 && x <= 0` to fail. An `&&` was surely meant. It changes nothing, because
// the caller only ever passes `inuclRndm()` in [0,1), but a port that "fixes" it to `&&` has the
// same behaviour and a port that reads it as a range check has the wrong idea of the domain.
//
// **2. The non-equilibrium evaporator's exciton walk has a Newton solve inside a rejection loop
// inside a retry loop.** `X` is found by solving `X^QEX = R` to 1% by Newton iteration, capped at
// a thousand steps and with NO convergence test other than `|DX/X| < 0.01`; the whole thing sits
// inside `while (EEXS_new < 0 && itry < 1000)` which sits inside `while (itry1 < 1000 && ...)`.
// The draw count of one call is therefore a strong function of the exciton configuration, which
// is exactly what makes it a good oracle column.
//
// **3. `G4FissionStore::generateConfiguration` reads one past the end.** Its walk is
// `while (configProbs[igen] <= st && igen < size()) igen++;` - the subscript is evaluated BEFORE
// the bound is tested, and `configurations[igen]` afterwards is unguarded too. `st` is
// `totProb * rand` with `rand` in [0,1), so it normally stays below the last cumulative entry;
// rounding at `rand` very close to 1 can put it above. Refused by name rather than reproduced:
// what it reads is not defined by the source.
//
// **4. The fissioner's `potentialMinimization` is a four-dimensional Newton minimisation with a
// 4x4 matrix, run up to two thousand times per candidate split, and it is called for EVERY
// candidate - up to fifty per fission.** It draws no random numbers, so it is deterministic and
// exactly comparable; it is also, by a wide margin, the most arithmetic in the package.
#ifndef G4GPU_BERTINI_DEEXCITE_CUH
#define G4GPU_BERTINI_DEEXCITE_CUH

#include <cmath>
#include "data/g4pow.hh"
#include "physics/hadronic/bertini/cascade_interpolator.cuh"
#include "physics/hadronic/bertini/collision_output.cuh"
#include "physics/hadronic/bertini/workspace.cuh"

namespace g4gpu::physics::hadronic::bert {

/// What the de-excitation chain could not do.
enum class DeexciteRefusal : int {
  kNone = 0,
  kBigBangTooManyNucleons,   ///< more nucleons than BertiniWorkspace::kMaxBangA
  kBigBangFailed,            ///< generateBangInSCM exhausted its thousand attempts
  kFissionStoreOffEnd,       ///< generateConfiguration walked past its last entry - see note 3
  kOverflow,                 ///< an output-list capacity
  kDeexStackOverflow,        ///< more fission levels than kMaxDeexStack
  kLevelDensityAboveTable    ///< nucleiLevelDensity was asked for A > 245
};

/// Which of those is THIS PORT declining and which is Geant4 itself producing nothing.
///
/// `kBigBangFailed` is Geant4's: `generateBangInSCM` gives up after a thousand attempts,
/// `G4BigBanger::deExcite` prints "No bang! Don't know why..." and RETURNS, leaving the fragment
/// out of the output entirely - with its own comment admitting "This will violate baryon number,
/// momentum, energy, etc." It is not rare under a prescribed engine: every A = 10 case in
/// `bertini_deexcite.csv` ends that way after 42,666 deviates. A caller that treated it as a
/// refusal would abort events Geant4 completes, so it is reported and stepped over, exactly as
/// the collider's `kKinematicsFailed` is.
__host__ __device__ inline bool deexcite_refusal_is_port_limit(DeexciteRefusal r) {
  return r != DeexciteRefusal::kNone && r != DeexciteRefusal::kBigBangFailed;
}

/// `G4CascadeDeexciteBase::getTargetData` - what every module in this file starts from.
///
/// `PEX` is the fragment's four-momentum in BERTINI's GeV and `EEXS` its excitation in Geant4's
/// MeV, which is the unit mismatch that runs through the whole chain: nearly every line that
/// mixes them has a `/GeV` or a `*GeV` on it, and the ones that do not are the bugs.
struct DeexTarget {
  int a = 0;
  int z = 0;
  LV pex;            ///< GeV
  double eexs = 0.0; ///< MeV
};

__host__ __device__ inline DeexTarget deex_target_data(const deex::Fragment& f) {
  DeexTarget t;
  t.a = f.a;
  t.z = f.z;
  t.pex = LV(f.momentum.v * 0.001, f.momentum.e * 0.001);
  t.eexs = f.excitation;
  return t;
}

/// `G4CascadeDeexciteBase::makeFragment(mom, A, Z, EX)`.
///
/// The four-momentum is FORCED onto the mass shell of `getNucleiMass(A,Z) + EX/GeV` - the
/// three-momentum survives and the energy is rebuilt - and the exciton counts are zeroed. So a
/// fragment handed on from one module to the next carries no memory of how it was excited.
__host__ __device__ inline deex::Fragment deex_make_fragment(const LV& mom, int a, int z,
                                                             double ex_MeV) {
  const double mass = inucl_nuclei_mass(a, z) + ex_MeV * 0.001;
  const LV m = lv_set_vect_m(mom.v, mass);
  deex::Fragment f;
  f.set_za_and_momentum(LV(m.v * 1000.0, m.e * 1000.0), z, a);
  return f;
}

__host__ __device__ inline deex::Fragment deex_make_fragment(int a, int z, double ex_MeV) {
  return deex_make_fragment(LV(), a, z, ex_MeV);
}

/// G4CascadeDeexciteBase::explosion - the BASE class's rule, which is what
/// G4CascadeDeexcitation asks. `Z == 0` with no upper bound on A is deliberate: a ball of
/// neutrons has nothing to evaporate through a Coulomb barrier and is dispersed instead.
__host__ __device__ inline bool deex_explosion_base(int a, int z, double excitation) {
  const int a_cut = 20;
  const double be_cut = 3.0;
  return (a <= a_cut || z == 0) && (excitation >= be_cut * inucl_binding_energy(a, z));
}

/// G4EquilibriumEvaporator::explosion - the DERIVED rule, "different criteria from base class,
/// since nucleus more agitated". Note `z >= 0` inside the negation is always true, so the test
/// is really `!(a >= 12 && z < 3*(a-z))`.
__host__ __device__ inline bool deex_explosion_equilibrium(int a, int z, double excitation) {
  const double be_cut = 3.0;
  return !(a >= 12 && z >= 0 && z < 3 * (a - z)) &&
         (excitation >= be_cut * inucl_binding_energy(a, z));
}

// =============================================================================================
// G4BigBanger
// =============================================================================================

/// G4BigBanger::xProbability - the single-nucleon energy-fraction density.
///
/// The parity of A selects between two forms of the phase-space weight, and the guard above the
/// body is `if (x < 1.0 || x > 0.0)`, an OR that no argument can fail. See note 1 in the header.
__host__ __device__ inline double bang_x_probability(double x, int a) {
  double ekpr = 0.0;
  if (x < 1.0 || x > 0.0) {
    ekpr = x * x;
    if (a % 2 == 0) {
      ekpr *= std::sqrt(1.0 - x) * g4pow_n(1.0 - x, (3 * a - 6) / 2);
    } else {
      ekpr *= g4pow_n(1.0 - x, (3 * a - 5) / 2);
    }
  }
  return ekpr;
}

__host__ __device__ inline double bang_max_probability(int a) {
  return bang_x_probability(2.0 / 3.0 / (double(a) - 1.0), a);
}

/// G4BigBanger::generateX - rejection against `promax`, a thousand tries, and on failure it
/// returns `maxProbability(a)` - a PROBABILITY where an energy fraction is wanted. Transcribed
/// as written; the fallback is what Geant4 hands to `generateMomentumModules` on exhaustion.
template <typename Rng>
__host__ __device__ inline double bang_generate_x(int a, double promax, Rng& rng) {
  const int itry_max = 1000;
  int itry = 0;
  while (itry < itry_max) {
    ++itry;
    const double x = rng.uniform();
    if (bang_x_probability(x, a) >= promax * rng.uniform()) { return x; }
  }
  return bang_max_probability(a);
}

/// G4BigBanger::generateMomentumModules. For A = 2 the split is exactly half and half and no
/// deviate is drawn; above that each nucleon gets a rejection-sampled fraction and they are
/// renormalised to the available energy afterwards.
template <typename Rng>
__host__ __device__ inline void bang_generate_momentum_modules(double etot, int a, int z,
                                                               double* mod, Rng& rng) {
  const double mp = inucl_particle_mass(kProton);
  const double mn = inucl_particle_mass(kNeutron);
  double xtot = 0.0;
  if (a > 2) {
    const double promax = bang_max_probability(a);
    for (int i = 0; i < a; ++i) {
      mod[i] = bang_generate_x(a, promax, rng);
      xtot += mod[i];
    }
  } else {
    xtot = 1.0;
    mod[0] = 0.5;
    mod[1] = 0.5;
  }
  for (int i = 0; i < a; ++i) {
    const double mass = (i < z) ? mp : mn;
    mod[i] *= etot / xtot;
    mod[i] = std::sqrt(mod[i] * (mod[i] + 2.0 * mass));
  }
}

/// G4BigBanger::generateBangInSCM - the momenta, back to back for A = 2 and with the last two
/// solved for by momentum conservation above that.
///
/// The three-vectors here are stored in four-vector objects whose energy component is NOT the
/// energy of anything - Geant4's own comments say "This is only a three-vector, not a
/// four-vector" four times - and `scm_momentums.push_back(-mom)` for the deuteron case negates
/// all four components of one of them. It does not matter, because `particles[i].fill(mom, knd)`
/// puts every one of them onto a NUCLEON's mass shell afterwards and the energy is discarded.
/// The same shape as the ion-projectile `initializeCascad` (cascade_model.cuh).
///
/// Returns the number of nucleons filled, or 0 on exhaustion.
template <typename Rng>
__host__ __device__ inline int bang_generate_in_scm(double etot, int a, int z, double* mod,
                                                    LV* scm, Rng& rng) {
  const double ang_cut = 0.9999;
  const int itry_max = 1000;

  if (a == 1) {
    // "Special -- bare nucleon doesn't really explode": zero momentum, and no deviate.
    scm[0] = LV();
    return 1;
  }

  bool bad = true;
  int itry = 0;
  int n = 0;
  while (bad && itry < itry_max) {
    ++itry;
    n = 0;
    bang_generate_momentum_modules(etot, a, z, mod, rng);
    if (a == 2) {
      const LV mom = inucl_with_random_angles(mod[0], 0.0, rng);
      scm[0] = mom;
      scm[1] = LV(Vec3d{-mom.v.x, -mom.v.y, -mom.v.z}, -mom.e);
      n = 2;
      bad = false;
    } else {
      LV tot_mom;
      for (int i = 0; i < a - 2; ++i) {
        const LV mom = inucl_with_random_angles(mod[i], 0.0, rng);
        scm[n++] = mom;
        tot_mom += mom;
      }
      const double tot_mod = g4gpu::mag(tot_mom.v);
      const double ct = -0.5 *
                        (tot_mod * tot_mod + mod[a - 2] * mod[a - 2] -
                         mod[a - 1] * mod[a - 1]) /
                        tot_mod / mod[a - 2];
      if (std::fabs(ct) < ang_cut) {
        const LV mom2 = inucl_with_fixed_theta(ct, mod[a - 2], 0.0, rng);
        // Rotate into the frame whose z axis is the accumulated momentum. Written out exactly
        // as Geant4 writes it, including the division by `a_tr` which is singular when the
        // accumulated momentum is along z - a state no random draw reaches but which a
        // rewrite into `rotateUz` would smooth over.
        const Vec3d apr = tot_mom.v * (1.0 / tot_mod);
        const double a_tr = std::sqrt(apr.x * apr.x + apr.y * apr.y);
        LV mom;
        mom.v.x = mom2.v.z * apr.x + (mom2.v.x * apr.y + mom2.v.y * apr.z * apr.x) / a_tr;
        mom.v.y = mom2.v.z * apr.y + (-mom2.v.x * apr.x + mom2.v.y * apr.z * apr.y) / a_tr;
        mom.v.z = mom2.v.z * apr.z - mom2.v.y * a_tr;
        scm[n++] = mom;
        scm[n++] = LV(Vec3d{-mom.v.x - tot_mom.v.x, -mom.v.y - tot_mom.v.y,
                            -mom.v.z - tot_mom.v.z},
                      -mom.e - tot_mom.e);
        bad = false;
      }
    }
  }
  return bad ? 0 : n;
}

/// G4BigBanger::deExcite - disperse the whole fragment into free nucleons.
///
/// `etot = (EEXS - bindingEnergy(A,Z))/1000`, floored at zero: the excitation minus what it
/// costs to unbind everything. The nucleons are thrown in the fragment's rest frame and boosted
/// out with `PEX.boostVector()`, and `PEX` includes the excitation because `getTargetData` reads
/// the fragment's own four-momentum.
template <typename Rng>
__host__ __device__ inline void bang_deexcite(const deex::Fragment& target,
                                              CollisionOutput& out, BertiniWorkspace& ws,
                                              Rng& rng, DeexciteRefusal& refusal) {
  const DeexTarget t = deex_target_data(target);
  if (t.a > BertiniWorkspace::kMaxBangA) {
    ws.overflow = CascadeOverflow::kBigBangNucleons;
    refusal = DeexciteRefusal::kBigBangTooManyNucleons;
    return;
  }
  const Vec3d to_lab = t.pex.boost_vector();

  double etot = (t.eexs - inucl_binding_energy(t.a, t.z)) * 0.001;
  if (etot < 0.0) { etot = 0.0; }

  const int n = bang_generate_in_scm(etot, t.a, t.z, ws.bang_modules, ws.bang_momenta, rng);
  if (n == 0) {
    // "No bang! Don't know why..." - Geant4 prints and RETURNS, leaving the fragment out of the
    // output entirely, with its own comment admitting "This will violate baryon number,
    // momentum, energy, etc." Reported by name rather than reproduced as a silent hole.
    refusal = DeexciteRefusal::kBigBangFailed;
    return;
  }

  // `particles[i].fill(scm_momentums[i], knd)` puts each three-vector on a nucleon's mass shell,
  // then the boost to the lab, then `setMomentum` stores again.
  for (int i = 0; i < n; ++i) {
    const int knd = (i < t.z) ? kProton : kNeutron;
    LV mom = inucl_store_momentum(ws.bang_momenta[i], knd);
    mom.boost(to_lab);
    mom = inucl_store_momentum(mom, knd);
    ws.bang_momenta[i] = mom;
  }
  // std::sort(particles, G4ParticleLargerEkin()) before they reach the output. The sort has to
  // move the TYPE with the momentum, and the type is a function of the INDEX (the first Z are
  // protons), so a parallel array is built first.
  int* kinds = ws.bang_kinds;
  double* ekin = ws.bang_ekin;
  for (int i = 0; i < n; ++i) {
    kinds[i] = (i < t.z) ? kProton : kNeutron;
    ekin[i] = inucl_stored_kinetic_energy(ws.bang_momenta[i], kinds[i]);
  }
  for (int i = 1; i < n; ++i) {
    const LV km = ws.bang_momenta[i];
    const int kk = kinds[i];
    const double ke = ekin[i];
    int j = i - 1;
    while (j >= 0 && ekin[j] < ke) {
      ws.bang_momenta[j + 1] = ws.bang_momenta[j];
      kinds[j + 1] = kinds[j];
      ekin[j + 1] = ekin[j];
      --j;
    }
    ws.bang_momenta[j + 1] = km;
    kinds[j + 1] = kk;
    ekin[j + 1] = ke;
  }
  for (int i = 0; i < n; ++i) {
    if (!co_add_particle(out, kinds[i], ws.bang_momenta[i])) {
      refusal = DeexciteRefusal::kOverflow;
      return;
    }
  }
}

// =============================================================================================
// G4NonEquilibriumEvaporator
// =============================================================================================

__host__ __device__ inline double noneq_matrix_element(int a) {
  return (a > 150) ? 100.0 : (a > 20) ? 140.0 : 70.0;
}
__host__ __device__ inline double noneq_e0() { return 200.0; }
__host__ __device__ inline double noneq_par_lev(int a) { return 0.125 * double(a); }

/// G4NonEquilibriumEvaporator::deExcite - the exciton walk.
///
/// Each iteration either emits a nucleon (N -> N-1), adds an exciton pair (N -> N+2), or stops.
/// Which of the three is a three-way choice between `D[0]` (the transition width), `D[1]`
/// (neutron escape) and `D[2]` (proton escape), and the walk continues only while one of the
/// escape widths is above `width_cut = 0.005` times the transition width.
///
/// The escaping particle's kinetic energy comes from solving `X^QEX = R` for X - the fraction of
/// the available energy left behind - by Newton iteration when QEX > 2 and in closed form
/// (`1 - sqrt(R)`) when QEX == 2. The iteration is capped at a thousand steps with a 1% relative
/// convergence test and no fallback: whatever X it has after the loop is used.
template <typename Rng>
__host__ __device__ inline void noneq_deexcite(const deex::Fragment& target,
                                               const ExitonConfiguration& ex,
                                               CollisionOutput& out, Rng& rng,
                                               DeexciteRefusal& refusal) {
  const int a_cut = 5;
  const int z_cut = 3;
  const double eexs_cut = 0.1;
  const double coul_coeff = 1.4;
  const int itry_max = 1000;
  const double width_cut = 0.005;

  DeexTarget t = deex_target_data(target);
  const LV pin = t.pex;

  // `G4ExitonConfiguration config(target)` reads the counts OFF the G4Fragment; P3's Fragment
  // has no room for them, so they arrive beside it. See CollisionOutput::recoil_excitons.
  int qpp = ex.proton_quasi_particles;
  int qnp = ex.neutron_quasi_particles;
  int qph = ex.proton_holes;
  int qnh = ex.neutron_holes;

  int qp = qpp + qnp;
  int qh = qph + qnh;
  int qex = qp + qh;

  LorentzConvertor to_exiton_rest;
  to_exiton_rest.bullet =
      LV(Vec3d{0.0, 0.0, std::sqrt(1.0e-6 * (1.0e-6 + 2.0 * inucl_particle_mass(kProton)))},
         1.0e-6 + inucl_particle_mass(kProton));

  const double efn = inucl_fermi_energy(t.a, t.z, 0);
  const double efp = inucl_fermi_energy(t.a, t.z, 1);

  int ar = t.a - qp;
  int zr = t.z - qpp;
  int nex = qex;
  LV ppout;
  bool try_again = (nex > 0);
  double esp = 0.0;
  int n_emitted = 0;

  while (try_again) {
    if (!(t.a >= a_cut && t.z >= z_cut && t.eexs > eexs_cut)) { try_again = false; break; }

    const double nuc_mass = inucl_nuclei_mass(t.a, t.z, t.eexs);
    t.pex = lv_set_vect_m(t.pex.v, nuc_mass);
    to_exiton_rest.target = t.pex;
    lc_to_the_target_rest_frame(to_exiton_rest);

    const double mel = noneq_matrix_element(t.a);
    const double e0 = noneq_e0();
    const double pl = noneq_par_lev(t.a);
    const double parlev = pl / double(t.a);
    const double eg = pl * t.eexs;

    if (!(double(qex) < std::sqrt(2.0 * eg))) { try_again = false; break; }

    double ak1 = 0.0, cpa1 = 0.0;
    para_maker_get_truncated(double(t.z), ak1, cpa1);

    const double vp = coul_coeff * double(t.z) * ak1 / (inucl_cbrt_int(t.a - 1) + 1.0) /
                      (1.0 + t.eexs / e0);
    const double dm1 = inucl_binding_energy(t.a, t.z);
    const double bn = dm1 - inucl_binding_energy(t.a - 1, t.z);
    const double bp = dm1 - inucl_binding_energy(t.a - 1, t.z - 1);
    const double emn = t.eexs - bn;
    const double emp = t.eexs - bp - vp * double(t.a) / double(t.a - 1);

    if (!(emn > eexs_cut)) { try_again = false; break; }

    int icase = 0;
    if (nex > 1) {
      const double aph = 0.25 * (double(qp) * qp + double(qh) * qh + qp - 3.0 * qh);
      const double aph1 = aph + 0.5 * (qp + qh);
      esp = t.eexs / double(qex);
      double mele = mel / esp / (double(t.a) * t.a * t.a);

      if (esp > 15.0) {
        mele *= std::sqrt(15.0 / esp);
      } else if (esp < 7.0) {
        mele *= std::sqrt(esp / 7.0);
        if (esp < 2.0) { mele *= std::sqrt(esp / 2.0); }
      }

      const double f1 = eg - aph;
      const double f2 = eg - aph1;
      if (f1 > 0.0 && f2 > 0.0) {
        const double f = f2 / f1;
        const double m1 = 2.77 * mele * pl;
        double d[3] = {0.0, 0.0, 0.0};
        d[0] = m1 * f2 * f2 * g4pow_n(f, nex - 1) / double(qex + 1);
        if (d[0] > 0.0) {
          if (nex >= 2) {
            d[1] = 0.0462 / parlev / inucl_cbrt_int(t.a) * double(qp) * t.eexs / double(qex);
            if (emp > eexs_cut) {
              d[2] = d[1] * g4pow_n(emp / t.eexs, nex) * (1.0 + cpa1);
            }
            d[1] *= g4pow_n(emn / t.eexs, nex) * inucl_get_al(t.a);
            if (qnp < 1) { d[1] = 0.0; }
            if (qpp < 1) { d[2] = 0.0; }

            try_again = nex > 1 && (d[1] > width_cut * d[0] || d[2] > width_cut * d[0]);
            if (try_again) {
              const double d5 = d[0] + d[1] + d[2];
              const double sl = d5 * rng.uniform();
              double s1 = 0.0;
              for (int i = 0; i < 3; ++i) {
                s1 += d[i];
                if (sl <= s1) { icase = i; break; }
              }
            }
          }
        } else {
          try_again = false;
        }
      } else {
        try_again = false;
      }
    }
    if (!try_again) { break; }

    if (icase > 0) {                      // N -> N-1, a nucleon escapes
      double v = 0.0;
      int ptype = 0;
      double b = 0.0;
      if (t.a < 3) { try_again = false; }

      if (try_again) {
        if (icase == 1) {                 // neutron
          if (qnp < 1) { icase = 0; }
          else { b = bn; v = 0.0; ptype = kNeutron; }
        } else {                          // proton
          if (qpp < 1) { icase = 0; }
          else {
            b = bp;
            v = vp;
            ptype = kProton;
            if (t.z - 1 < 1) { try_again = false; }
          }
        }

        if (try_again && icase != 0) {
          const double eb = t.eexs - b;
          const double e = eb - v * double(t.a) / double(t.a - 1);
          if (e < 0.0) {
            icase = 0;
          } else {
            const double e1 = eb - v;
            int itry1 = 0;
            bool bad = true;
            while (itry1 < itry_max && icase > 0 && bad) {
              ++itry1;
              int itry = 0;
              double eexs_new = -1.0;
              double epart = 0.0;
              while (eexs_new < 0.0 && itry < itry_max) {
                ++itry;
                const double r = rng.uniform();
                double x;
                if (qex == 2) {
                  x = 1.0 - std::sqrt(r);
                } else {
                  const double qex2 = 1.0 / double(qex);
                  const double qex1 = 1.0 / double(qex - 1);
                  x = g4gpu::data::g4pow_pow_a(0.5 * r, qex2);
                  for (int i = 0; i < 1000; ++i) {
                    const double dx =
                        x * qex1 * (1.0 + qex2 * x * (1.0 - r / g4pow_n(x, nex)) / (1.0 - x));
                    x -= dx;
                    if (std::fabs(dx / x) < 0.01) { break; }
                  }
                }
                epart = eb - x * e1;
                eexs_new = eb - epart * double(t.a) / double(t.a - 1);
              }
              if (itry == itry_max || eexs_new < 0.0) { icase = 0; continue; }

              epart *= 0.001;             // MeV -> GeV
              const double mass = inucl_particle_mass(ptype);
              const double pmod = std::sqrt(epart * (2.0 * mass + epart));
              LV mom = inucl_with_random_angles(pmod, mass, rng);
              mom = lc_back_to_the_lab(to_exiton_rest, mom);

              int qpp_new = qpp;
              int qnp_new = qnp;
              int a_new = t.a - 1;
              int z_new = t.z;
              if (ptype == kProton) { --qpp_new; --z_new; }
              if (ptype == kNeutron) { --qnp_new; }

              const double mass_new = inucl_nuclei_mass(a_new, z_new);
              eexs_new = ((t.pex - mom).mag() - mass_new) * 1000.0;
              if (eexs_new < 0.0) { continue; }   // "Sanity check for new nucleus"

              t.pex -= mom;
              t.eexs = eexs_new;
              t.a = a_new;
              t.z = z_new;
              --nex;
              --qex;
              --qp;
              qpp = qpp_new;
              qnp = qnp_new;

              mom = inucl_store_momentum(mom, ptype);
              if (!co_add_particle(out, ptype, mom)) {
                refusal = DeexciteRefusal::kOverflow;
                return;
              }
              ++n_emitted;
              ppout += mom;
              bad = false;
            }
            if (itry1 == itry_max) { icase = 0; }
          }
        }
      }
    }

    if (icase == 0 && try_again) {        // N -> N+2, an exciton pair is created
      const double tnn = 1.6 * efn + esp;
      const double tnp = 1.6 * efp + esp;
      const double xnun = 1.0 / (1.6 + esp / efn);
      const double xnup = 1.0 / (1.6 + esp / efp);
      const double snn1 = inucl_cs_nn(tnp) * xnup;
      const double snn2 = inucl_cs_nn(tnn) * xnun;
      const double spn1 = inucl_cs_pn(tnp) * xnup;
      const double spn2 = inucl_cs_pn(tnn) * xnun;
      const double pp = (double(qpp) * snn1 + double(qnp) * spn1) * double(zr);
      const double pn = (double(qpp) * spn2 + double(qnp) * snn2) * double(ar - zr);
      const double pw = pp + pn;
      nex += 2;
      qex += 2;
      ++qp;
      ++qh;
      --ar;
      if (ar > 1) {
        const double sl = pw * rng.uniform();
        if (sl > pp) {
          ++qnp;
          ++qnh;
        } else {
          ++qpp;
          ++qph;
          --zr;
          if (zr < 2) { try_again = false; }
        }
      } else {
        try_again = false;
      }
    }
  }

  // The residual. When nothing was emitted the INPUT fragment is handed on unchanged - not a
  // copy rebuilt from (A, Z, EEXS), which would have lost the exciton configuration the
  // equilibrium stage does not read but `makeRecoilFragment` put there.
  if (n_emitted == 0) {
    out.recoil_fragment = target;
    out.has_recoil_fragment = true;
  } else {
    out.recoil_fragment = deex_make_fragment(pin - ppout, t.a, t.z, t.eexs);
    out.has_recoil_fragment = true;
  }
}

// =============================================================================================
// G4Fissioner
// =============================================================================================

/// The 72-point fission-barrier table `getQF` interpolates, and its abscissa.
__host__ __device__ inline const double* fission_qfrep() {
  static const double v[72] = {
      22.5, 22.0, 21.0, 21.0, 20.0, 20.6, 20.6, 18.6, 15.8, 13.5, 6.5,
      6.65, 6.22, 6.27, 6.5,  6.7,  6.2,  6.25, 5.9,  6.1,  5.75,
      6.46, 5.7,  6.28, 5.8,  6.15, 5.6,  5.8,  5.2,  5.8,
      6.2,  5.9,  5.9,  6.0,  5.8,  5.7,  5.4,  5.4,
      5.6,  6.1,  5.57, 6.3,  5.5,  5.8,  4.7,  6.2,  6.4,  6.2,
      6.5,  6.2,  6.5,  5.3,  6.4,  5.7,  5.7,  6.2,  5.7,
      6.3,  5.8,  6.7,  5.8,  6.6,  6.1,  4.3,
      6.2,  3.8,  5.6,  4.0,  4.0,  4.2,  4.2,  3.5};
  return v;
}
__host__ __device__ inline const double* fission_xrep() {
  static const double v[72] = {
      0.6761, 0.677,  0.6788, 0.6803, 0.685,
      0.6889, 0.6914, 0.6991, 0.7068, 0.725,  0.7391,
      0.74,   0.741,  0.742,  0.743,  0.744,  0.7509, 0.752, 0.7531, 0.7543, 0.7548,
      0.7557, 0.7566, 0.7576,
      0.7587, 0.7597, 0.7608, 0.762,  0.7632, 0.7644, 0.7675, 0.7686, 0.7697, 0.7709,
      0.7714, 0.7721, 0.7723, 0.7733, 0.7743, 0.7753, 0.7764,
      0.7775, 0.7786, 0.7801, 0.781,  0.7821, 0.7831, 0.7842, 0.7852,
      0.7864, 0.7875, 0.7880, 0.7887, 0.7889, 0.7899, 0.7909, 0.7919, 0.7930,
      0.7941, 0.7953, 0.7965, 0.7977, 0.7987, 0.7989,
      0.7997, 0.8075, 0.8097, 0.8119, 0.8143, 0.8164, 0.8174, 0.8274};
  return v;
}

__host__ __device__ inline double fission_c2(int a1, int a2, double x3, double x4,
                                             double r12) {
  return 124.57 * (1.0 / double(a1) + 1.0 / double(a2)) + 0.78 * (x3 + x4) -
         176.9 * ((x3 * x3 * x3 * x3) + (x4 * x4 * x4 * x4)) +
         219.36 * (1.0 / (double(a1) * a1) + 1.0 / (double(a2) * a2)) - 1.108 / r12;
}

__host__ __device__ inline double fission_zopt(int a1, int a2, int zt, double x3, double x4,
                                               double r12) {
  return (87.7 * (x4 - x3) * (1.0 - 1.25 * (x4 + x3)) +
          double(zt) * ((124.57 / double(a2) + 0.78 * x4 - 176.9 * (x4 * x4 * x4 * x4) +
                         219.36 / (double(a2) * a2)) -
                        0.554 / r12)) /
         fission_c2(a1, a2, x3, x4, r12);
}

/// G4Fissioner::potentialMinimization - a four-dimensional Newton minimisation of the
/// deformation energy of the two nascent fragments, over their two quadrupole and two hexadecapole
/// parameters, with a steepest-descent step scaled by `ST/ST1`.
///
/// `AL1` and `BET1` are IN-OUT: the caller seeds them with (-0.15, 0.05) once and every
/// subsequent candidate split starts from where the previous one converged. So the fifty
/// candidates of one fission are not independent, and evaluating them in a different order would
/// give different answers.
__host__ __device__ inline void fission_potential_minimization(double& vp, double (&ed)[2],
                                                               double& vc, int af, int as,
                                                               int zf, int zs,
                                                               double (&al1)[2],
                                                               double (&bet1)[2],
                                                               double& r12) {
  const double huge_num = 2.0e35;
  const int itry_max = 2000;
  const double dsol1 = 1.0e-6;
  const double ds1 = 0.3;
  const double ds2 = 1.0 / ds1 / ds1;
  const int a1[2] = {af, as};
  const int z1[2] = {zf, zs};
  const double d = 1.01844 * double(zf) * double(zs);
  const double d0 = 1.0e-3 * d;
  double r[2], c[2], f[2];
  r12 = 0.0;

  for (int i = 0; i < 2; ++i) {
    r[i] = inucl_cbrt_int(a1[i]);
    const double y1 = r[i] * r[i];
    const double y2 = double(z1[i]) * z1[i] / r[i];
    c[i] = 6.8 * y1 - 0.142 * y2;
    f[i] = 12.138 * y1 - 0.145 * y2;
  }

  double sal[2], sbe[2], x[2], xx1[2], xx2[2], ral[2], rbe[2];
  double aa[4][4], b[4];
  int itry = 0;
  while (itry < itry_max) {
    ++itry;
    double s = 0.0;
    for (int i = 0; i < 2; ++i) {
      s += r[i] * (1.0 + al1[i] + bet1[i] - 0.257 * al1[i] * bet1[i]);
    }
    r12 = 0.0;
    double y1 = 0.0, y2 = 0.0;
    for (int i = 0; i < 2; ++i) {
      sal[i] = r[i] * (1.0 - 0.257 * bet1[i]);
      sbe[i] = r[i] * (1.0 - 0.257 * al1[i]);
      x[i] = r[i] / s;
      xx1[i] = x[i] * x[i];
      xx2[i] = x[i] * xx1[i];
      y1 += al1[i] * xx1[i];
      y2 += bet1[i] * xx2[i];
      r12 += r[i] * (1.0 - al1[i] * (1.0 - 0.6 * x[i]) + bet1[i] * (1.0 - 0.429 * xx1[i]));
    }
    const double y3 = -0.6 * y1 + 0.857 * y2;
    const double y4 = (1.2 * y1 - 2.571 * y2) / s;
    const double r2 = d0 / (r12 * r12);
    const double r3 = 2.0 * r2 / r12;

    for (int i = 0; i < 2; ++i) {
      ral[i] = -r[i] * (1.0 - 0.6 * x[i]) + sal[i] * y3;
      rbe[i] = r[i] * (1.0 - 0.429 * xx1[i]) + sbe[i] * y3;
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        const double del1 = (i == j) ? 1.0 : 0.0;
        double dx1 = 0.0, dx2 = 0.0;
        if (std::fabs(al1[i]) >= ds1) {
          const double xxx = al1[i] * al1[i] * ds2;
          const double dex = (xxx > 100.0) ? huge_num : std::exp(xxx);
          dx1 = 2.0 * (1.0 + 2.0 * al1[i] * al1[i] * ds2) * dex * ds2;
        }
        if (std::fabs(bet1[i]) >= ds1) {
          const double xxx = bet1[i] * bet1[i] * ds2;
          const double dex = (xxx > 100.0) ? huge_num : std::exp(xxx);
          dx2 = 2.0 * (1.0 + 2.0 * bet1[i] * bet1[i] * ds2) * dex * ds2;
        }
        const double del = 2.0e-3 * del1;
        // NOTE the asymmetry Geant4 has here: the [i][j] block is built from RBE, not RAL,
        // even though its derivative terms are the alpha ones. Transcribed as written.
        aa[i][j] = r3 * rbe[i] * rbe[j] -
                   r2 * (-0.6 * (xx1[i] * sal[j] + xx1[j] * sal[i]) + sal[i] * sal[j] * y4) +
                   del * c[i] + del1 * dx1;
        const int i1 = i + 2;
        const int j1 = j + 2;
        aa[i1][j1] = r3 * rbe[i] * rbe[j] -
                     r2 * (0.857 * (xx2[i] * sbe[j] + xx2[j] * sbe[i]) +
                           sbe[i] * sbe[j] * y4) +
                     del * f[i] + del1 * dx2;
        aa[i][j1] = r3 * ral[i] * rbe[j] -
                    r2 * (0.857 * (xx2[j] * sal[i] - 0.6 * xx1[i] * sbe[j]) +
                          sbe[j] * sal[i] * y4 - 0.257 * r[i] * y3 * del1);
        aa[j1][i] = aa[i][j1];
      }
    }

    for (int i = 0; i < 2; ++i) {
      double dx1 = 0.0, dx2 = 0.0;
      if (std::fabs(al1[i]) >= ds1) { dx1 = 2.0 * al1[i] * ds2 * std::exp(al1[i] * al1[i] * ds2); }
      if (std::fabs(bet1[i]) >= ds1) {
        dx2 = 2.0 * bet1[i] * ds2 * std::exp(bet1[i] * bet1[i] * ds2);
      }
      b[i] = r2 * ral[i] - 2.0e-3 * c[i] * al1[i] + dx1;
      b[i + 2] = r2 * rbe[i] - 2.0e-3 * f[i] * bet1[i] + dx2;
    }

    double st = 0.0, st1 = 0.0;
    for (int i = 0; i < 4; ++i) {
      st += b[i] * b[i];
      for (int j = 0; j < 4; ++j) { st1 += aa[i][j] * b[i] * b[j]; }
    }
    const double step = st / st1;
    double dsol = 0.0;
    for (int i = 0; i < 2; ++i) {
      al1[i] += b[i] * step;
      bet1[i] += b[i + 2] * step;
      dsol += b[i] * b[i] + b[i + 2] * b[i + 2];
    }
    dsol = std::sqrt(dsol);
    if (dsol < dsol1) { break; }
  }

  for (int i = 0; i < 2; ++i) { ed[i] = f[i] * bet1[i] * bet1[i] + c[i] * al1[i] * al1[i]; }
  vc = d / r12;
  vp = vc + ed[0] + ed[1];
}

/// G4Fissioner::deExcite. Returns the number of fragments written (0 or 2).
///
/// `G4FissionStore`'s candidate list lives in the workspace, five parallel arrays rather than
/// one array of structs - as locals the fifty candidates cost 2 kB of a thread's stack frame,
/// and `G4FissionConfiguration::epot` is stored by Geant4 and never read back, so it is not
/// kept at all. The fifty is exact, not an estimate: the filling loop is
/// `for (i = 0; i < 50 && A1 > 30; i++)`.
template <typename Rng>
__host__ __device__ inline int fission_deexcite(const deex::Fragment& target,
                                                deex::Fragment* frag_out,
                                                BertiniWorkspace& ws, Rng& rng,
                                                DeexciteRefusal& refusal) {
  const DeexTarget t = deex_target_data(target);
  const double a13 = inucl_cbrt_int(t.a);
  const double mass_in = t.pex.mag();
  const double e_in = mass_in;      // "Mass includes excitation"

  double para = 0.055 * a13 * a13 * (inucl_cbrt_int(t.a - t.z) + inucl_cbrt_int(t.z));
  double tem = std::sqrt(t.eexs / para);
  double teta = 0.494 * a13 * tem;
  teta = teta / std::sinh(teta);

  if (t.a < 246) {
    bool bad_nld = false;
    const double nld = inucl_nuclei_level_density(t.a, bad_nld);
    if (bad_nld) { refusal = DeexciteRefusal::kLevelDensityAboveTable; return 0; }
    para += (nld - para) * teta;
  }

  int a1 = t.a / 2 + 1;
  int a2 = t.a - a1;
  double alma = -1000.0;
  const double dm1 = inucl_binding_energy(t.a, t.z);
  const double evv = t.eexs - dm1;
  const double dm2 = inucl_binding_energy_asymptotic(t.a, t.z);
  const double dtem = (t.a < 220) ? 0.5 : 1.15;
  tem += dtem;

  double al1[2] = {-0.15, -0.15};
  double bet1[2] = {0.05, 0.05};
  double r12 = inucl_cbrt_int(a1) + inucl_cbrt_int(a2);

  int n_cfg = 0;

  for (int i = 0; i < 50 && a1 > 30; ++i) {
    --a1;
    a2 = t.a - a1;
    const double x3 = 1.0 / inucl_cbrt_int(a1);
    const double x4 = 1.0 / inucl_cbrt_int(a2);
    // G4lrint is a round-to-nearest, and the `- 1.` is applied BEFORE it.
    int z1 = int(std::floor(fission_zopt(a1, a2, t.z, x3, x4, r12) - 1.0 + 0.5));
    int z2 = t.z - z1;
    double edef1[2];
    double vpot = 0.0, vcoul = 0.0;
    fission_potential_minimization(vpot, edef1, vcoul, a1, a2, z1, z2, al1, bet1, r12);

    const double dm3 = inucl_binding_energy(a1, z1);
    const double dm4 = inucl_binding_energy_asymptotic(a1, z1);
    const double dm5 = inucl_binding_energy(a2, z2);
    const double dm6 = inucl_binding_energy_asymptotic(a2, z2);
    const double dmt1 = dm4 + dm6 - dm2;
    const double dmt = dm3 + dm5 - dm1;
    const double ezl = t.eexs + dmt - vpot;

    if (ezl > 0.0) {
      const double c1 = std::sqrt(fission_c2(a1, a2, x3, x4, r12) / tem);
      double dz = inucl_random_gauss(c1, rng);
      dz = (dz > 0.0) ? dz + 0.5 : -std::fabs(dz - 0.5);
      z1 += int(dz);
      z2 -= int(dz);

      const double defin = inucl_random_gauss(tem, rng);
      const double ez = (dmt1 + (dmt - dmt1) * teta - vpot + defin) / tem;
      if (ez >= alma) { alma = ez; }
      const double ek = vcoul + defin + 0.5 * tem;
      const double ev = evv + inucl_binding_energy(a1, z1) + inucl_binding_energy(a2, z2) - ek;
      if (ev > 0.0 && n_cfg < BertiniWorkspace::kMaxFissionConfigs) {
        ws.fission_afirst[n_cfg] = double(a1);
        ws.fission_zfirst[n_cfg] = double(z1);
        ws.fission_ezet[n_cfg] = ez;
        ws.fission_ekin[n_cfg] = ek;
        ++n_cfg;
      }
    }
  }

  if (n_cfg == 0) { return 0; }

  // G4FissionStore::generateConfiguration - a Boltzmann weight in `ezet - amax`, floored at
  // exp(-30), and a cumulative walk. See note 3 in the header: Geant4's walk subscripts before
  // it bounds-checks, so the off-the-end case is refused here rather than reproduced.
  const double small = -30.0;
  double tot_prob = 0.0;
  for (int i = 0; i < n_cfg; ++i) {
    double pr = ws.fission_ezet[i] - alma;
    if (pr < small) { pr = small; }
    pr = std::exp(pr);
    tot_prob += pr;
    ws.fission_probs[i] = tot_prob;
  }
  const double st = tot_prob * rng.uniform();
  int igen = 0;
  while (igen < n_cfg && ws.fission_probs[igen] <= st) { ++igen; }
  if (igen >= n_cfg) { refusal = DeexciteRefusal::kFissionStoreOffEnd; return 0; }

  a1 = int(ws.fission_afirst[igen]);
  a2 = t.a - a1;
  int z1 = int(ws.fission_zfirst[igen]);
  int z2 = t.z - z1;

  const double mass1 = inucl_nuclei_mass(a1, z1);
  const double mass2 = inucl_nuclei_mass(a2, z2);
  const double ek = ws.fission_ekin[igen];
  const double pmod = std::sqrt(0.001 * ek * mass1 * mass2 / mass_in);

  const LV mom1 = inucl_with_random_angles(pmod, mass1, rng);
  const LV mom2 = lv_set_vect_m(Vec3d{-mom1.v.x, -mom1.v.y, -mom1.v.z}, mass2);

  const double e_out = mom1.e + mom2.e;
  const double ev = 1000.0 * (e_in - e_out) / double(t.a);
  if (ev <= 0.0) { return 0; }        // "No fission energy"

  frag_out[0] = deex_make_fragment(mom1, a1, z1, ev * a1);
  frag_out[1] = deex_make_fragment(mom2, a2, z2, ev * a2);
  return 2;
}

// =============================================================================================
// G4EquilibriumEvaporator
// =============================================================================================

__host__ __device__ inline bool eq_good_remnant(int a, int z) {
  return a > 1 && z > 0 && a > z;
}
__host__ __device__ inline double eq_parlevden() { return 0.125; }
__host__ __device__ inline double eq_e0() { return 200.0; }

/// G4EquilibriumEvaporator::getQF - the fission barrier. Inside the tabulated range of the
/// fissility parameter it interpolates the 72-point table; outside, a liquid-drop formula.
__host__ __device__ inline double eq_get_qf(double x, double x2, int a) {
  const double g0 = 20.4;
  const double xmin = 0.6761;
  const double xmax = 0.8274;
  double qff;
  if (x < xmin || x > xmax) {
    const double x1 = 1.0 - 0.02 * x2;
    const double fx = (0.73 + (3.33 * x1 - 0.66) * x1) * (x1 * x1 * x1);
    const double a13 = inucl_cbrt_int(a);
    qff = g0 * fx * a13 * a13;
  } else {
    qff = interp_value(x, fission_xrep(), fission_qfrep(), 72);
  }
  if (qff < 0.0) { qff = 0.0; }
  return qff;
}

/// "ugly parameterisation to fit the experimental fission cs for Hg - Bi nuclei" - Geant4's own
/// description, and it depends on nothing but the excitation.
__host__ __device__ inline double eq_get_af(double e) {
  double af = 1.285 * (1.0 - e / 1100.0);
  if (af < 1.06) { af = 1.06; }
  return af;
}

/// One pass of G4EquilibriumEvaporator::deExcite over ONE fragment, with fission fragments
/// pushed onto `ws.deex_stack` rather than recursed into. Returns false on a refusal.
template <typename Rng>
__host__ __device__ inline bool eq_deexcite_one(const deex::Fragment& target,
                                                CollisionOutput& out, BertiniWorkspace& ws,
                                                Rng& rng, DeexciteRefusal& refusal) {
  const double huge_num = 50.0;
  const double small = -50.0;
  const double prob_cut_off = 1.0e-15;
  const double q1[6] = {0.0, 0.0, 2.23, 8.49, 7.72, 28.3};
  const int an[6] = {1, 1, 2, 3, 3, 4};
  const int qq[6] = {0, 1, 1, 1, 2, 2};
  const double gg[6] = {2.0, 2.0, 6.0, 6.0, 6.0, 4.0};
  const double be = 0.0063;
  const double fission_cut = 1000.0;
  const double cut_off_energy = 0.1;
  const double bf = 0.0242;
  const int itry_max = 1000;
  const int itry_global_max = 1000;
  const int itry_gam_max = 100;

  DeexTarget t = deex_target_data(target);

  double w[8];
  double u[6], v[6], tm[6];
  int aa1[6], zz1[6];

  LorentzConvertor to_nuclei_rest;
  to_nuclei_rest.bullet =
      LV(Vec3d{0.0, 0.0, std::sqrt(1.0e-6 * (1.0e-6 + 2.0 * inucl_particle_mass(kProton)))},
         1.0e-6 + inucl_particle_mass(kProton));

  LV ppout;

  if (deex_explosion_equilibrium(t.a, t.z, t.eexs)) {
    bang_deexcite(target, out, ws, rng, refusal);
    return !deexcite_refusal_is_port_limit(refusal);
  }
  if (t.eexs < cut_off_energy) {
    out.recoil_fragment = target;
    out.has_recoil_fragment = true;
    return true;
  }

  const double coul_coeff = (t.a >= 100) ? 1.4 : 1.2;
  const LV pin = t.pex;
  bool try_again = true;
  bool fission_open = true;
  int itry_global = 0;

  while (try_again && itry_global < itry_global_max) {
    ++itry_global;
    to_nuclei_rest.target = t.pex;
    lc_to_the_target_rest_frame(to_nuclei_rest);

    if (deex_explosion_equilibrium(t.a, t.z, t.eexs)) {
      bang_deexcite(deex_make_fragment(t.pex, t.a, t.z, t.eexs), out, ws, rng, refusal);
      return !deexcite_refusal_is_port_limit(refusal);
    }
    if (t.eexs < cut_off_energy) { try_again = false; break; }

    const double e0 = eq_e0();
    const double parlev = eq_parlevden();
    const double u1 = parlev * t.a;
    const ParaMakerParams pm = para_maker_get_params(double(t.z));
    const double dm0 = inucl_binding_energy(t.a, t.z);

    for (int i = 0; i < 6; ++i) {
      aa1[i] = t.a - an[i];
      zz1[i] = t.z - qq[i];
      u[i] = parlev * aa1[i];
      v[i] = 0.0;
      tm[i] = -0.1;
      if (eq_good_remnant(aa1[i], zz1[i])) {
        const double qb = dm0 - inucl_binding_energy(aa1[i], zz1[i]) - q1[i];
        v[i] = coul_coeff * double(t.z) * qq[i] * pm.ak[i] / (1.0 + t.eexs / e0) /
               (inucl_cbrt_int(aa1[i]) + inucl_cbrt_int(an[i]));
        tm[i] = t.eexs - qb - v[i] * double(t.a) / double(aa1[i]);
      }
    }

    const double ue = 2.0 * std::sqrt(u1 * t.eexs);
    double prob_sum = 0.0;

    w[0] = 0.0;
    if (tm[0] > cut_off_energy) {
      const double al = inucl_get_al(t.a);
      const double a13 = inucl_cbrt_int(aa1[0]);
      w[0] = be * a13 * a13 * gg[0] * al;
      double tm1 = 2.0 * std::sqrt(u[0] * tm[0]) - ue;
      if (tm1 > huge_num) { tm1 = huge_num; } else if (tm1 < small) { tm1 = small; }
      w[0] *= std::exp(tm1);
      prob_sum += w[0];
    }
    for (int i = 1; i < 6; ++i) {
      w[i] = 0.0;
      if (tm[i] > cut_off_energy) {
        const double a13 = inucl_cbrt_int(aa1[i]);
        w[i] = be * a13 * a13 * gg[i] * (1.0 + pm.cpa[i]);
        double tm1 = 2.0 * std::sqrt(u[i] * tm[i]) - ue;
        if (tm1 > huge_num) { tm1 = huge_num; } else if (tm1 < small) { tm1 = small; }
        w[i] *= std::exp(tm1);
        prob_sum += w[i];
      }
    }

    w[6] = 0.0;
    if (t.a >= 100 && fission_open) {
      // **`X2 = Z * Z / A` is INTEGER division**, and it is the whole fission channel.
      //
      // `A` and `Z` are `G4int` members of G4CascadeDeexciteBase, so `Z*Z/A` truncates: lead-207
      // gives 32, not 32.4831. The line below it, `X1 = 1.0 - 2.0*Z/A`, has a double literal in
      // front and does NOT truncate - the two are adjacent and only one of them is integer
      // arithmetic. `X2` then feeds the fissility parameter
      // `x = 0.019316*X2/(1 - 1.79*X1^2)`, which for Pb-207 comes out 0.6698 instead of 0.6799 -
      // and 0.6761 is `getQF`'s XMIN. So the truncation moves the nucleus from INSIDE the
      // 72-point barrier table to OUTSIDE it, and the barrier is computed by the liquid-drop
      // formula instead: 30.8 MeV rather than the table's 21.0.
      //
      // The fission width is exponential in `2*sqrt(AF*(EEXS-QF))`, so a 10 MeV barrier is a
      // factor of TWENTY-TWO: w[6] = 3.1e-4 with the truncation and 6.9e-3 without. MEASURED as
      // a change of outcome, not of digits - with the floating-point form the port fissioned a
      // 300 MeV lead residual where Geant4 chose an alpha channel, exhausted its sampler and
      // returned the nucleus untouched. Every other quantity in that case agreed.
      const double x2 = double(t.z * t.z / t.a);
      const double x1 = 1.0 - 2.0 * double(t.z) / double(t.a);
      const double x = 0.019316 * x2 / (1.0 - 1.79 * x1 * x1);
      const double ef = t.eexs - eq_get_qf(x, x2, t.a);
      if (ef > 0.0) {
        const double af = u1 * eq_get_af(t.eexs);
        double tm1 = 2.0 * std::sqrt(af * ef) - ue;
        if (tm1 > huge_num) { tm1 = huge_num; } else if (tm1 < small) { tm1 = small; }
        w[6] = bf * std::exp(tm1);
        // The fission width is capped at a thousand times the NEUTRON width - and if the neutron
        // channel is closed, w[0] is zero and the cap is zero, so fission is switched off with
        // it. That coupling is not obvious and it is what the `fission_cut*W[0]` line does.
        if (w[6] > fission_cut * w[0]) { w[6] = fission_cut * w[0]; }
        prob_sum += w[6];
      }
    }

    int icase = -1;

    if (prob_sum < prob_cut_off) {          // nothing can be emitted: the photon chain
      const double ucr0 = 2.5 + 150.0 / double(t.a);
      const double t00 = 1.0 / (std::sqrt(u1 / ucr0) - 1.25 / ucr0);
      int itry_gam = 0;
      while (t.eexs > cut_off_energy && try_again) {
        ++itry_gam;
        int itry = 0;
        const double t04 = 4.0 * t00;
        double fmax;
        if (t04 < t.eexs) {
          fmax = (t04 * t04 * t04 * t04) * std::exp((t.eexs - t04) / t00);
        } else {
          fmax = t.eexs * t.eexs * t.eexs * t.eexs;
        }
        double s = 0.0;
        while (itry < itry_max) {
          ++itry;
          s = t.eexs * rng.uniform();
          const double x1 = (s * s * s * s) * std::exp((t.eexs - s) / t00);
          if (x1 > fmax * rng.uniform()) { break; }
        }
        if (itry == itry_max) { try_again = false; break; }

        if (s < t.eexs) {
          s *= 0.001;                        // MeV -> GeV
          LV mom = inucl_with_random_angles(s, 0.0, rng);
          mom = lc_back_to_the_lab(to_nuclei_rest, mom);
          t.pex -= mom;
          t.eexs -= s * 1000.0;
          mom = inucl_store_momentum(mom, kPhoton);
          if (!co_add_particle(out, kPhoton, mom)) {
            refusal = DeexciteRefusal::kOverflow;
            return false;
          }
          ppout += mom;
        } else {
          if (itry_gam == itry_gam_max) { try_again = false; }
        }
      }
      try_again = false;
    } else {
      const double sl = prob_sum * rng.uniform();
      double s1 = 0.0;
      for (int i = 0; i < 7; ++i) {
        s1 += w[i];
        if (sl <= s1) { icase = i; break; }
      }
      if (icase < 0) { continue; }

      if (icase < 6) {                        // a nucleon or a light ion escapes
        const double xmax =
            (std::sqrt(u[icase] * tm[icase] + 0.25) - 0.5) / u[icase];
        int itry1 = 0;
        bool bad = true;
        while (itry1 < itry_max && bad) {
          ++itry1;
          int itry = 0;
          double s = 0.0;
          while (itry < itry_max) {
            ++itry;
            // Dostrovsky eq. 17, sampled by rejection.
            const double x = rng.uniform() * tm[icase];
            const double ptest =
                (x / xmax) * std::exp(-2.0 * u[icase] * xmax +
                                      2.0 * std::sqrt(u[icase] * (tm[icase] - x)));
            if (rng.uniform() < ptest) { s = x + v[icase]; break; }
          }

          if (s > v[icase] && s < t.eexs) {
            s *= 0.001;                       // MeV -> GeV
            if (icase < 2) {
              const int ptype = 2 - icase;    // icase 0 -> neutron (2), 1 -> proton (1)
              const double mass = inucl_particle_mass(ptype);
              const double pmod = std::sqrt((2.0 * mass + s) * s);
              LV mom = inucl_with_random_angles(pmod, mass, rng);
              mom = lc_back_to_the_lab(to_nuclei_rest, mom);
              const double mass_new = inucl_nuclei_mass(aa1[icase], zz1[icase]);
              const double eexs_new = ((t.pex - mom).mag() - mass_new) * 1000.0;
              if (eexs_new < 0.0) { continue; }
              t.pex -= mom;
              t.eexs = eexs_new;
              t.a = aa1[icase];
              t.z = zz1[icase];
              mom = inucl_store_momentum(mom, ptype);
              if (!co_add_particle(out, ptype, mom)) {
                refusal = DeexciteRefusal::kOverflow;
                return false;
              }
              ppout += mom;
              bad = false;
            } else {
              const double mass = inucl_nuclei_mass(an[icase], qq[icase]);
              const double pmod = std::sqrt((2.0 * mass + s) * s);
              LV mom = inucl_with_random_angles(pmod, mass, rng);
              mom = lc_back_to_the_lab(to_nuclei_rest, mom);
              const double mass_new = inucl_nuclei_mass(aa1[icase], zz1[icase]);
              const double eexs_new = ((t.pex - mom).mag() - mass_new) * 1000.0;
              if (eexs_new < 0.0) { continue; }
              t.pex -= mom;
              t.eexs = eexs_new;
              t.a = aa1[icase];
              t.z = zz1[icase];
              const InuclNucleus nn = nuclei_fill(mom, an[icase], qq[icase], 0.0);
              if (!co_add_nucleus(out, nn)) {
                refusal = DeexciteRefusal::kOverflow;
                return false;
              }
              ppout += nn.momentum;
              bad = false;
            }
          }
        }
        if (itry1 == itry_max || bad) { try_again = false; }
      } else {                                // fission
        deex::Fragment ff[2];
        const int nf =
            fission_deexcite(deex_make_fragment(t.a, t.z, t.eexs), ff, ws, rng, refusal);
        if (deexcite_refusal_is_port_limit(refusal)) { return false; }
        if (nf == 2) {
          // "Move fission fragments to lab frame for processing" - through the same converter
          // the evaporation used, reflection and rotation included.
          for (int i = 0; i < 2; ++i) {
            const LV in(ff[i].momentum.v * 0.001, ff[i].momentum.e * 0.001);
            bool und = false;
            const LV lab = co_boost_one(in, to_nuclei_rest, und);
            ff[i].set_momentum(LV(lab.v * 1000.0, lab.e * 1000.0));
          }
          // Geant4 recurses on fragment 0 and then on fragment 1. A stack drained from the top
          // does the same when 1 is pushed before 0.
          if (ws.n_deex_stack + 2 > BertiniWorkspace::kMaxDeexStack) {
            ws.overflow = CascadeOverflow::kDeexStack;
            refusal = DeexciteRefusal::kDeexStackOverflow;
            return false;
          }
          ws.deex_stack[ws.n_deex_stack++] = ff[1];
          ws.deex_stack[ws.n_deex_stack++] = ff[0];
          return true;
        }
        fission_open = false;               // "fission forbidden now"
      }
    }
  }

  // What is left is a NUCLEUS on the outgoing list, not a recoil fragment - the equilibrium
  // evaporator is the end of the chain and there is nothing after it to de-excite.
  const LV pnuc = pin - ppout;
  if (!co_add_nucleus(out, nuclei_fill(pnuc, t.a, t.z, 0.0))) {
    refusal = DeexciteRefusal::kOverflow;
    return false;
  }
  return true;
}

/// G4EquilibriumEvaporator::deExcite, with the fission recursion flattened onto a stack.
template <typename Rng>
__host__ __device__ inline void eq_deexcite(const deex::Fragment& target, CollisionOutput& out,
                                            BertiniWorkspace& ws, Rng& rng,
                                            DeexciteRefusal& refusal) {
  ws.n_deex_stack = 0;
  if (!eq_deexcite_one(target, out, ws, rng, refusal)) { return; }
  // A fission fragment whose own big bang failed is Geant4 returning early from ONE recursive
  // call; the other fragment is still processed. So the stack is drained regardless.
  while (ws.n_deex_stack > 0) {
    const deex::Fragment f = ws.deex_stack[--ws.n_deex_stack];
    if (!eq_deexcite_one(f, out, ws, rng, refusal)) { return; }
  }
}

// =============================================================================================
// G4CascadeDeexcitation
// =============================================================================================

/// G4CascadeDeexcitation::deExcite - the whole chain.
///
/// Big bang if the fragment qualifies under the BASE class's rule, otherwise non-equilibrium
/// evaporation followed by equilibrium evaporation of whatever it leaves. The intermediate
/// output is split: the non-equilibrium ejectiles go to the caller's list, its RECOIL becomes
/// the equilibrium stage's input, and the equilibrium stage's whole output (particles, light
/// ions and the final nucleus) is added wholesale.
template <typename Rng>
__host__ __device__ inline void cascade_deexcite(const deex::Fragment& fragment,
                                                 const ExitonConfiguration& excitons,
                                                 CollisionOutput& out, CollisionOutput& tmp,
                                                 BertiniWorkspace& ws, Rng& rng,
                                                 DeexciteRefusal& refusal) {
  refusal = DeexciteRefusal::kNone;
  if (deex_explosion_base(fragment.a, fragment.z, fragment.excitation)) {
    bang_deexcite(fragment, out, ws, rng, refusal);
    return;
  }

  co_reset(tmp);
  noneq_deexcite(fragment, excitons, tmp, rng, refusal);
  if (deexcite_refusal_is_port_limit(refusal)) { return; }

  // Only the PARTICLES are copied on; the recoil is the next stage's input.
  for (int i = 0; i < tmp.n_particles; ++i) {
    if (!co_add_particle(out, tmp.particles[i].type, tmp.particles[i].momentum)) {
      refusal = DeexciteRefusal::kOverflow;
      return;
    }
  }
  const deex::Fragment newfrag = tmp.recoil_fragment;

  co_reset(tmp);
  eq_deexcite(newfrag, tmp, ws, rng, refusal);
  if (deexcite_refusal_is_port_limit(refusal)) { return; }
  co_add(out, tmp);
}

}  // namespace g4gpu::physics::hadronic::bert

#endif  // G4GPU_BERTINI_DEEXCITE_CUH
