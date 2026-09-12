// G4Fancy3DNucleus - the nucleus every intra-nuclear model in QBBC is built on.
//
// Transcribed from G4Fancy3DNucleus.{hh,cc}, G4Fancy3DNucleusHelper.hh (hadronic/util) and
// G4V3DNucleus.hh in 11.1.1, with CLHEP's RandGauss::shoot (externals/clhep/src/RandGauss.cc)
// for the one branch that needs a Gaussian.
//
// It arranges A nucleons in position and momentum: `ChooseNucleons` decides which are protons,
// `ChoosePositions` places them by rejection against a nuclear density with a hard-core
// exclusion, `ChooseFermiMomenta` gives each a momentum inside the local Fermi sphere, and
// `ReduceSum` then forces the total three-momentum to zero by editing the components parallel
// to the residual sum. Both `G4BinaryCascade` (through `G4BinaryLightIonReaction`) and
// `G4FTFModel` construct one, which is why this file is under `bic/nucleus/` with a contract
// header (`nucleus_model.cuh`) rather than inside the cascade.
//
// SEVEN things about the Geant4 class are load-bearing and easy to lose in a transcription.
//
// 1. **Every nucleon comes out OFF its mass shell, downwards.** `ChooseFermiMomenta`'s last
//    loop sets `energy = GetPDGMass() - BindingEnergy()/myA` and pairs it with a sampled Fermi
//    three-momentum, so `Get4Momentum().mag()` is `sqrt(E^2 - p^2)` with E already below the
//    PDG mass - well below it, by ~8 MeV per nucleon plus the Fermi kinetic energy. That is
//    what makes `G4GeneratorPrecompoundInterface::Propagate` select its QGS arm for any nucleus
//    straight out of `Init`, and that arm dereferences a null primary projectile.
//    docs/RISK.md V50. Any caller that hands this nucleus to `preco::propagate_residual` must
//    pass a primary four-momentum or be refused.
//
// 2. **The retry loop around `ChooseFermiMomenta` runs once.** It is written
//    `for (G4int ntry=0; ntry<1; ntry++) { ...; if (ReduceSum()) break; }`, so the `break` is
//    unreachable-as-a-loop-control and `ReduceSum`'s return value is discarded. A configuration
//    whose momenta cannot be balanced is used anyway, with a non-zero total three-momentum and
//    no message. The port returns the verdict in `NucleusReport::reduce_sum_failed` instead of
//    discarding it, because the only other way to find out is to add up the momenta.
//
// 3. **A = 12 is a different nucleus.** Carbon-12 is placed as three alpha clusters on the
//    corners of an equilateral triangle (Bozek et al., Phys. Rev. C90, 064902 (2014)), the
//    hard-core distance is raised from 0.8 fm to 0.9 fm, and `CenterNucleons()` is called for
//    it and for nothing else. Since QBBC's most important target for this project is carbon,
//    the special case is not a corner - it is the main line.
//
// 4. **The C12 cluster spread is a variance used as a standard deviation.** `Disp = 0.552`
//    with the comment `// 0.91^2*2/3 fermi^2`, and `G4RandGauss::shoot(0., Disp)`'s second
//    argument is a STANDARD DEVIATION (RandGauss.icc:25, `shoot()*stdDev + mean`). So the
//    sampled displacement has sigma = 0.552 fm where the quoted parametrisation gives
//    sigma = sqrt(0.552) = 0.743 fm - the clusters come out 26% tighter than the paper's.
//    Reproduced as written; docs/RISK.md V68.
//
// 5. **The C12 branch's random stream depends on a process-wide latch.** CLHEP's
//    `RandGauss::shoot` generates two deviates per polar-method trial and caches the second in
//    a `CLHEP_THREAD_LOCAL` static (`set_st`, `nextGauss_st`). So whether the FIRST Gaussian of
//    a C12 `Init` consumes two uniforms or none depends on how many Gaussians the process drew
//    earlier - anywhere, in any model. The port keeps the latch in the scratch struct so that a
//    caller that reuses one scratch reproduces the sequence, and a caller that does not gets a
//    fresh latch; which of the two matches Geant4 depends on what else the run did.
//    docs/RISK.md V68.
//
// 6. **`ChoosePositions` draws from two streams, interleaved.** The trial position comes from a
//    block of `min(600, 9*(A-i))` uniforms filled by `G4RandFlat::shootArray` and consumed from
//    the TOP DOWNWARDS (`jx=--jr; jy=--jr; ... arand[--jr]`), while the density acceptance test
//    is a separate single `G4UniformRand()`. A block is refilled when fewer than three values
//    remain, and the leftovers are discarded. Reproducing the acceptance rate is not enough:
//    reproducing the stream needs the block, the direction, and the refill threshold.
//
// 7. **Two `DoLorentzBoost` overloads boost in opposite directions.** See `nucleon.cuh`.
//
// **What is refused, by name.** A hyper-nucleus's MASS (`myL > 0` reaches
// `G4HyperNucleiProperties::GetNuclearMass`, which P3's handler refuses for the same reason);
// anti-nuclei, which have no `SetParticleType` path here and are P1's refused set; and the
// nucleon-array capacity, which is the caller's and is reported rather than truncated. The
// `NON_INTEGER_A_Z` entry point is `#if defined`-ed out in 11.1.1 and is not ported; its one
// line of behaviour (a fractional A rounded by a uniform draw) is recorded in `init` instead.
// Geant4's three FatalExceptions - the position loop running out of attempts, and the two C12
// cluster loops - become named refusals, because a kernel cannot throw.
#ifndef G4GPU_BIC_FANCY_3D_NUCLEUS_CUH
#define G4GPU_BIC_FANCY_3D_NUCLEUS_CUH

#include <cmath>

#include "core/units.cuh"
#include "core/vec3.cuh"
#include "data/g4pow.hh"
#include "physics/hadronic/bic/nucleus/fermi_momentum.cuh"
#include "physics/hadronic/bic/nucleus/nuclear_density.cuh"
#include "physics/hadronic/bic/nucleus/nucleon.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::bic {

/// The largest `9*(A-i)` block `G4RandFlat::shootArray` is asked for in `ChoosePositions`:
/// `G4double arand[600]` is a fixed local in Geant4 and `std::min(600, ...)` is the clamp.
inline constexpr int kFlatBlock = 600;

/// G4Fancy3DNucleusHelper - a (vector, size, index) triple sorted on `size`.
struct NucleusSortEntry {
  Vec3d vector{0.0, 0.0, 0.0};
  double size = 0.0;
  int index = 0;
};

/// What `Init` refused, and what it had to report.
///
/// `reduce_sum_failed` and `proton_momentum_zeroed` are REPORTS, not refusals: Geant4 continues
/// in both cases (the first silently, the second with a JustWarning) and so does this, because
/// reproducing Geant4 is the job. The other four stop the build and leave the nucleus empty.
struct NucleusReport {
  bool capacity = false;               ///< A exceeds the caller's nucleon array
  bool hyper_nucleus = false;          ///< myL > 0 reached GetMass
  bool placement_failed = false;       ///< ChoosePositions ran out of 1000*A attempts
  bool cluster_placement_failed = false;  ///< one of the three C12 alpha loops gave up
  bool reduce_sum_logic = false;       ///< ReduceSum's `best < 0` G4HadronicException
  bool reduce_sum_failed = false;      ///< ReduceSum returned false; Geant4 ignores this
  int proton_momentum_zeroed = 0;      ///< ChooseFermiMomenta's JustWarning, counted
  int refused_a = 0, refused_z = 0;

  __host__ __device__ bool fatal() const {
    return capacity || placement_failed || cluster_placement_failed || reduce_sum_logic;
  }
};

/// Scratch `Init` needs and the nucleus does not keep: G4Fancy3DNucleus holds these as data
/// members so it can reuse their allocations, and they carry no state between calls.
///
/// The two Gaussian fields are the exception and are deliberately NOT scratch: they mirror
/// CLHEP's `RandGauss::set_st` / `nextGauss_st` thread-local statics, so reusing one scratch
/// across `init` calls is what reproduces a Geant4 run's stream. See note 5 in the header.
struct Nucleus3DScratch {
  Vec3d* momentum = nullptr;           ///< [capacity] G4Fancy3DNucleus::momentum
  double* fermi_p = nullptr;           ///< [capacity] G4Fancy3DNucleus::fermiM
  NucleusSortEntry* test_sums = nullptr;  ///< [capacity] G4Fancy3DNucleus::testSums
  double* flat_block = nullptr;        ///< [kFlatBlock] the G4RandFlat::shootArray buffer
  int capacity = 0;

  bool gauss_cached = false;           ///< CLHEP RandGauss::set_st
  double gauss_next = 0.0;             ///< CLHEP RandGauss::nextGauss_st
};

/// CLHEP::RandGauss::shoot() (RandGauss.cc:62), the polar (Marsaglia) method, with its cache.
///
/// Returns `v2*fac` and caches `v1*fac`, in that order - not the other way round. Three
/// consequences, all observable: a call consumes either two-or-more uniforms or zero; the
/// FIRST of a pair returned is the second one computed; and the cache survives across calls.
template <typename Rng>
__host__ __device__ inline double rand_gauss_shoot(Nucleus3DScratch& s, Rng& rng) {
  if (s.gauss_cached) {
    s.gauss_cached = false;
    return s.gauss_next;
  }
  double r, v1, v2;
  do {
    v1 = 2.0 * rng.uniform() - 1.0;
    v2 = 2.0 * rng.uniform() - 1.0;
    r = v1 * v1 + v2 * v2;
  } while (r > 1.0);
  const double fac = std::sqrt(-2.0 * std::log(r) / r);
  s.gauss_next = v1 * fac;
  s.gauss_cached = true;
  return v2 * fac;
}

/// `G4RandGauss::shoot(mean, stdDev)` = `shoot()*stdDev + mean` (RandGauss.icc:25).
template <typename Rng>
__host__ __device__ inline double rand_gauss_shoot(Nucleus3DScratch& s, Rng& rng, double mean,
                                                   double std_dev) {
  return rand_gauss_shoot(s, rng) * std_dev + mean;
}

/// G4Fancy3DNucleus. The nucleon array is the caller's; everything else is by value.
struct Nucleus3D {
  Nucleon* nucleons = nullptr;
  int capacity = 0;

  int my_a = 0;
  int my_z = 0;
  int my_l = 0;
  int current = -1;                  ///< G4Fancy3DNucleus::currentNucleon
  NuclearDensity density;
  FermiMomentum fermi;
  double nucleondistance = 0.0;
  double excitation = 0.0;

  // ------------------------------------------------------------------------------------------
  // The accessors, all of them inline in G4Fancy3DNucleus.hh
  // ------------------------------------------------------------------------------------------

  __host__ __device__ int mass_number() const { return my_a; }
  __host__ __device__ int charge() const { return my_z; }
  __host__ __device__ int number_of_lambdas() const { return my_l; }
  __host__ __device__ double excitation_energy() const { return excitation; }
  __host__ __device__ double add_excitation_energy(double e) {
    excitation += e;
    return excitation;
  }

  /// G4Fancy3DNucleus::StartLoop / GetNextNucleon. The iterator is a member of the nucleus, so
  /// two consumers cannot walk one nucleus at the same time - a fact worth knowing before a
  /// kernel shares one.
  __host__ __device__ bool start_loop() {
    current = 0;
    return my_a > 0;
  }
  __host__ __device__ Nucleon* next_nucleon() {
    return (current >= 0 && current < my_a) ? &nucleons[current++] : nullptr;
  }

  /// G4Fancy3DNucleus::BindingEnergy = G4NucleiProperties::GetBindingEnergy(A, Z).
  __host__ __device__ double binding_energy() const { return deex::binding_energy(my_a, my_z); }

  /// G4Fancy3DNucleus::CoulombBarrier. `(1.44/1.14) MeV` is written as a division of two
  /// literals and kept that way: 1.263157894736842 is not what a four-digit decimal would give.
  __host__ __device__ double coulomb_barrier() const {
    const double cfactor = (1.44 / 1.14) * u::MeV<double>();
    return cfactor * static_cast<double>(my_z) / (1.0 + data::g4pow_z13<double>(my_a));
  }

  /// G4Fancy3DNucleus::GetNuclearRadius() and the overload it forwards to. The default relative
  /// density is 0.5, i.e. the HALF-density radius and not the r0 A^(1/3) radius.
  __host__ __device__ double nuclear_radius() const { return nuclear_radius(0.5); }
  __host__ __device__ double nuclear_radius(double max_relative_density) const {
    return density.radius(max_relative_density);
  }

  /// G4Fancy3DNucleus::GetOuterRadius - the furthest nucleon plus one hard-core distance. It
  /// is a property of the SAMPLED configuration, so it varies event to event; `G4BinaryLight-
  /// IonReaction` uses the sum of two of these as its impact-parameter range.
  __host__ __device__ double outer_radius() const {
    double maxradius2 = 0.0;
    for (int i = 0; i < my_a; ++i) {
      const double m2 = g4gpu::mag2(nucleons[i].position);
      if (m2 > maxradius2) { maxradius2 = m2; }
    }
    return std::sqrt(maxradius2) + nucleondistance;
  }

  /// G4Fancy3DNucleus::GetMass, the non-hyper branch.
  ///
  /// This is `Z m_p + (A-Z) m_n - GetBindingEnergy(A, Z)` and it is NOT
  /// `G4NucleiProperties::GetNuclearMass(A, Z)`, which for a tabulated nuclide returns the AME
  /// mass MINUS the atomic electron binding (`ame12_electron_mass`) and for Z <= 2 returns a
  /// PDG mass outright. The two differ by tens of eV for a medium nucleus - below any physics
  /// tolerance and above the one this port compares at - so both are kept and each is used
  /// where Geant4 uses it. `G4BinaryLightIonReaction` uses neither: it asks
  /// `G4IonTable::GetIonMass`, a third answer.
  __host__ __device__ double mass(NucleusReport& rep) const {
    if (my_l > 0) {
      rep.hyper_nucleus = true;
      rep.refused_a = my_a;
      rep.refused_z = my_z;
      return 0.0;
    }
    return static_cast<double>(my_z) * u::proton_mass_c2<double>() +
           static_cast<double>(my_a - my_z) * u::neutron_mass_c2<double>() - binding_energy();
  }

  // ------------------------------------------------------------------------------------------
  // The rigid-body operations
  // ------------------------------------------------------------------------------------------

  /// G4Fancy3DNucleus::DoLorentzBoost(const G4ThreeVector&).
  __host__ __device__ void do_lorentz_boost(const Vec3d& beta) {
    for (int i = 0; i < my_a; ++i) { nucleons[i].boost(beta); }
  }
  /// G4Fancy3DNucleus::DoLorentzBoost(const G4LorentzVector&). Opposite direction to the one
  /// above - see `nucleon.cuh`'s note on `G4Nucleon::Boost`.
  __host__ __device__ void do_lorentz_boost(const LorentzVector& p4) {
    for (int i = 0; i < my_a; ++i) { nucleons[i].boost(p4); }
  }

  /// G4Fancy3DNucleus::DoLorentzContraction(const G4ThreeVector&). Positions only.
  __host__ __device__ void do_lorentz_contraction(const Vec3d& beta) {
    const double beta2 = g4gpu::mag2(beta);
    if (beta2 > 0.0) {
      const double factor = (1.0 - std::sqrt(1.0 - beta2)) / beta2;  // (gamma-1)/gamma/beta^2
      for (int i = 0; i < my_a; ++i) {
        const Vec3d r = nucleons[i].position;
        nucleons[i].position = r - (factor * g4gpu::dot(beta, r)) * beta;
      }
    }
  }
  /// G4Fancy3DNucleus::DoLorentzContraction(const G4LorentzVector&). `theBoost.vect()/theBoost.e()`
  /// is `operator/(Hep3Vector, double)`, which is `v*(1.0/c)` (ThreeVector.cc:298) - the same
  /// reciprocal-then-multiply `boostVector()` does, so these two agree bit for bit. The
  /// distinction fragment.cuh's header records is between this and a component-wise division.
  __host__ __device__ void do_lorentz_contraction(const LorentzVector& p4) {
    if (p4.e != 0.0) {
      const double inv = 1.0 / p4.e;
      do_lorentz_contraction(Vec3d{p4.v.x * inv, p4.v.y * inv, p4.v.z * inv});
    }
  }

  /// G4Fancy3DNucleus::DoTranslation.
  __host__ __device__ void do_translation(const Vec3d& shift) {
    for (int i = 0; i < my_a; ++i) { nucleons[i].position = nucleons[i].position + shift; }
  }

  /// G4Fancy3DNucleus::CenterNucleons - shift so the mean of the POSITIONS is at the origin.
  /// `center /= -myA` then `DoTranslation(center)`, i.e. the shift is the negated mean.
  ///
  /// `Hep3Vector::operator/=(c)` is `*this *= 1.0/c` (ThreeVector.cc:307), a reciprocal and
  /// then three multiplications - not three divisions. Same arithmetic, different last bit.
  __host__ __device__ void center_nucleons() {
    Vec3d center{0.0, 0.0, 0.0};
    for (int i = 0; i < my_a; ++i) { center = center + nucleons[i].position; }
    const double inv = 1.0 / (-static_cast<double>(my_a));
    center = Vec3d{center.x * inv, center.y * inv, center.z * inv};
    do_translation(center);
  }

  /// G4Fancy3DNucleus::SortNucleonsIncZ / SortNucleonsDecZ.
  ///
  /// Geant4 uses `std::sort` on the position's z, which is unspecified for equal keys; an
  /// insertion sort is used here, which is stable. Two nucleons with bitwise-equal z is a
  /// measure-zero event for a sampled configuration and the only way to reach it is the
  /// zeroed-proton case in `ChooseFermiMomenta`, which sets `proton_momentum_zeroed` anyway -
  /// and that case zeroes a MOMENTUM, not a position. `SortNucleonsDecZ` sorts increasing and
  /// then reverses, which for a stable sort is not the same as sorting decreasing; the reverse
  /// is what Geant4 does.
  __host__ __device__ void sort_nucleons_inc_z() {
    if (my_a < 2) { return; }
    for (int i = 1; i < my_a; ++i) {
      Nucleon key = nucleons[i];
      int j = i - 1;
      while (j >= 0 && key.position.z < nucleons[j].position.z) {
        nucleons[j + 1] = nucleons[j];
        --j;
      }
      nucleons[j + 1] = key;
    }
  }
  __host__ __device__ void sort_nucleons_dec_z() {
    if (my_a < 2) { return; }
    sort_nucleons_inc_z();
    for (int i = 0, j = my_a - 1; i < j; ++i, --j) {
      Nucleon t = nucleons[i];
      nucleons[i] = nucleons[j];
      nucleons[j] = t;
    }
  }
};

// =============================================================================================
// The private implementation methods, in the order Init calls them
// =============================================================================================

/// G4Fancy3DNucleus::ChooseNucleons.
///
/// One uniform per iteration of a loop that may place NOTHING: if the draw lands in the proton
/// band but every proton is already placed, the iteration is spent and no nucleon is assigned.
/// So the number of deviates consumed is not A; it is a negative-binomial-ish count that runs
/// to about A + Z ln(A/(A-Z)) on average and is unbounded above. Reproducing it exactly is what
/// makes the rest of the stream line up.
///
/// The neutron branch's guard counts neutrons as `nucleons - protons - lambdas`, which is why
/// no neutron counter exists.
template <typename Rng>
__host__ __device__ inline void choose_nucleons(Nucleus3D& nuc, Rng& rng) {
  int protons = 0, nucleons = 0, lambdas = 0;
  const double prob_proton = static_cast<double>(nuc.my_z) / static_cast<double>(nuc.my_a);
  const double prob_lambda =
      (nuc.my_l > 0) ? static_cast<double>(nuc.my_l) / static_cast<double>(nuc.my_a) : 0.0;
  while (nucleons < nuc.my_a) {
    const double rnd = rng.uniform();
    if (rnd < prob_proton) {
      if (protons < nuc.my_z) {
        ++protons;
        nuc.nucleons[nucleons++].type = kProton;
      }
    } else if (rnd < prob_proton + prob_lambda) {
      if (lambdas < nuc.my_l) {
        ++lambdas;
        nuc.nucleons[nucleons++].type = kLambda;
      }
    } else {
      if ((nucleons - protons - lambdas) < (nuc.my_a - nuc.my_z - nuc.my_l)) {
        nuc.nucleons[nucleons++].type = kNeutron;
      }
    }
  }
}

/// G4Fancy3DNucleus::ChoosePositions, the `myA != 12` branch.
///
/// `maxR = GetNuclearRadius(0.001)` - the radius at one part in a thousand of the central
/// relative density. The comment beside it says "there are no nucleons at a relative Density of
/// 0.01", one decade off the number in the code; the code is what runs and 0.001 is what is
/// transcribed. (Section 8 of docs/HADRONIC_PLAN.md: a comment that justifies a discard is the
/// one to re-read. This one justifies a truncation with the wrong threshold.)
///
/// The proton extra test is the reason a proton's position distribution differs from a
/// neutron's in this model at all: a place whose LOCAL Fermi energy is at or below the nucleus's
/// Coulomb barrier is rejected for a proton, which empties the nuclear surface of protons. Note
/// that `CoulombBarrier()` is a whole-nucleus number and the Fermi energy is local, so the
/// rejected shell is `sqrt(pFermi^2 + m^2) - m <= 1.263 Z/(1 + A^(1/3)) MeV`.
///
/// Returns false if the 1000*A attempt budget ran out - Geant4's `mod_util001` FatalException.
template <typename Rng>
__host__ __device__ inline bool choose_positions(Nucleus3D& nuc, Nucleus3DScratch& sc,
                                                 Rng& rng) {
  int i = 0;
  const double nd2 = nuc.nucleondistance * nuc.nucleondistance;
  const double max_r = nuc.nuclear_radius(0.001);
  const double barrier = nuc.coulomb_barrier();
  int jr = 0;
  int iterations_left = 1000 * nuc.my_a;
  double* arand = sc.flat_block;

  while ((i < nuc.my_a) && (--iterations_left > 0)) {
    Vec3d a_pos{0.0, 0.0, 0.0};
    do {
      if (jr < 3) {
        jr = (kFlatBlock < 9 * (nuc.my_a - i)) ? kFlatBlock : 9 * (nuc.my_a - i);
        // G4RandFlat::shootArray(jr, prand) - jr uniforms into arand[0..jr-1], in order.
        for (int k = 0; k < jr; ++k) { arand[k] = rng.uniform(); }
      }
      const int jx = --jr;
      const int jy = --jr;
      const int jz = --jr;
      a_pos = Vec3d{2.0 * arand[jx] - 1.0, 2.0 * arand[jy] - 1.0, 2.0 * arand[jz] - 1.0};
    } while (g4gpu::mag2(a_pos) > 1.0);

    a_pos = a_pos * max_r;
    const double rel = nuc.density.relative_density(a_pos);
    if (rng.uniform() < rel) {
      bool freeplace = true;
      // Geant4 walks its own `places` vector, which holds exactly the positions of nucleons
      // 0..i-1 in that order and is cleared at the top of the method. Walking the nucleons
      // gives the same values in the same order, so the short-circuit stops at the same place.
      for (int k = 0; k < i && freeplace; ++k) {
        const Vec3d delta = nuc.nucleons[k].position - a_pos;
        freeplace = g4gpu::mag2(delta) > nd2;
      }
      if (freeplace) {
        const double p_fermi = nuc.fermi.fermi_momentum(nuc.density.density(a_pos));
        if (nuc.nucleons[i].type == kProton) {
          const double nuc_mass = nucleon_pdg_mass(kProton);
          const double e_fermi =
              std::sqrt(p_fermi * p_fermi + nuc_mass * nuc_mass) - nuc_mass;
          if (e_fermi <= barrier) { freeplace = false; }
        }
      }
      if (freeplace) {
        nuc.nucleons[i].position = a_pos;
        ++i;
      }
    }
  }
  return iterations_left > 0;
}

/// G4Fancy3DNucleus::ChoosePositions, the `myA == 12` branch: three alpha clusters at the
/// corners of an equilateral triangle of side 3.05 fm in the xy-plane, Gaussian-spread, then
/// the whole nucleus randomly rotated.
///
/// The three corners are (L/2, 0, 0), (-L/2, 0, 0) and (0, 0.866 L, 0) - `0.866` written out
/// rather than `sqrt(3)/2`, so the triangle is equilateral to four digits and not exactly.
/// The centroid of the three is therefore NOT at the origin, which is why `CenterNucleons()`
/// is called for A = 12 and for nothing else.
///
/// The FIRST nucleon of the first cluster is placed with no overlap test at all; nucleons 2-4
/// test against 0..i-1, and clusters two and three test against every nucleon placed so far.
/// Each cluster has its own 10,000-attempt budget, and the budget is NOT reset between the
/// nucleons of one cluster - it is a per-cluster total.
///
/// The rotation is `rotateZ(2 pi U)` then `rotateY(acos(2U - 1))`, applied as a LorentzRotation
/// to `(position, 0)`, so the time component stays zero and the three-vector is rotated by
/// R_y(theta) R_z(phi) - in that order, because `G4LorentzRotation::rotateY` post-multiplies.
template <typename Rng>
__host__ __device__ inline bool choose_positions_c12(Nucleus3D& nuc, Nucleus3DScratch& sc,
                                                     Rng& rng) {
  const double l_base = 3.05 * deex::fermi();
  const double disp = 0.552;  // "0.91^2*2/3 fermi^2" - a variance used as a sigma; see header
  const double nd2 = nuc.nucleondistance * nuc.nucleondistance;
  const Vec3d corner[3] = {Vec3d{l_base / 2.0, 0.0, 0.0}, Vec3d{-l_base / 2.0, 0.0, 0.0},
                           Vec3d{0.0, l_base * 0.866, 0.0}};

  // The first nucleon of the first cluster: no overlap test.
  {
    const Vec3d g{rand_gauss_shoot(sc, rng, 0.0, disp), rand_gauss_shoot(sc, rng, 0.0, disp),
                  rand_gauss_shoot(sc, rng, 0.0, disp)};
    nuc.nucleons[0].position = g * deex::fermi() + corner[0];
  }

  for (int cluster = 0; cluster < 3; ++cluster) {
    const int first = (cluster == 0) ? 1 : 4 * cluster;
    const int last = 4 * cluster + 4;
    int loop_counter_left = 10000;
    for (int ii = first; ii < last; ++ii) {
      bool cont;
      do {
        const Vec3d g{rand_gauss_shoot(sc, rng, 0.0, disp), rand_gauss_shoot(sc, rng, 0.0, disp),
                      rand_gauss_shoot(sc, rng, 0.0, disp)};
        nuc.nucleons[ii].position = g * deex::fermi() + corner[cluster];
        cont = false;
        for (int jj = 0; jj < ii; ++jj) {
          const Vec3d d = nuc.nucleons[ii].position - nuc.nucleons[jj].position;
          if (g4gpu::mag2(d) <= nd2) {
            cont = true;
            break;
          }
        }
      } while (cont && --loop_counter_left > 0);
    }
    if (loop_counter_left <= 0) { return false; }
  }

  // G4LorentzRotation RandomRotation; rotateZ(phi); rotateY(theta); then Pos *= RandomRotation.
  //
  // `HepLorentzVector::operator*=(const HepLorentzRotation&)` is `*this = m.vectorMultiplication
  // (*this)`, and HepRotation::rotateY POST-multiplies the existing matrix, so the composite is
  // R = R_z(phi) then R_y(theta) applied in that order to the vector: v -> R_y(R_z(v)) is what
  // `rotateZ(phi); rotateY(theta);` builds... no: `rotateZ` on an identity gives R_z, then
  // `rotateY` gives R_z * R_y? HepRotation::rotateY is `*this = HepRotationY(a) * (*this)`, i.e.
  // PRE-multiplication, so the matrix is R_y * R_z and the vector sees R_z first. Written out
  // component-wise below in exactly that order so there is nothing to get backwards.
  const double phi = u::twopi<double>() * rng.uniform();
  const double theta = std::acos(2.0 * rng.uniform() - 1.0);
  const double cp = std::cos(phi), sp = std::sin(phi);
  const double ct = std::cos(theta), st = std::sin(theta);
  for (int ii = 0; ii < nuc.my_a; ++ii) {
    const Vec3d p = nuc.nucleons[ii].position;
    // R_z(phi): x' = x cos - y sin, y' = x sin + y cos
    const double x1 = p.x * cp - p.y * sp;
    const double y1 = p.x * sp + p.y * cp;
    const double z1 = p.z;
    // R_y(theta): x'' = x cos + z sin, z'' = -x sin + z cos
    nuc.nucleons[ii].position = Vec3d{x1 * ct + z1 * st, y1, -x1 * st + z1 * ct};
  }
  return true;
}

/// G4Fancy3DNucleus::ReduceSum, as a loop rather than tail recursion.
///
/// The problem it solves: A independently sampled Fermi momenta do not sum to zero, and a
/// nucleus with net momentum would drift. The method zeroes the sum by REFLECTING individual
/// momenta about the plane perpendicular to the residual sum - `delta = 2 (p . u) u` subtracted
/// from p flips that component's sign - largest first, until the remainder is small enough that
/// the LAST nucleon can absorb it inside its own Fermi momentum.
///
/// Four details:
///   * `PFermi` is `fermiM[myA-1]`, the last nucleon's, re-read at the top of every recursion -
///     so a swap changes the target.
///   * the descending scan is `while ((sum - testSums[--index].Vector).mag() > PFermi && index>0)`,
///     which decrements BEFORE the test and checks `index > 0` after it, so it can exit with
///     index == 0 having tested entry 0.
///   * inside that loop a reflection is applied only `if (sum.mag() > (sum-delta).mag())`, i.e.
///     only when it does not overshoot - so the scan can pass over entries without using them.
///   * the closing search picks, among entries 0..index, the delta whose `|delta - sum|` is
///     closest to the last nucleon's ALREADY-SAMPLED momentum magnitude, and then assigns
///     `momentum[A-1] = delta - sum`. So the last nucleon's momentum is overwritten by an exact
///     balance, and the sampled value is used only as a preference.
///
/// Returns false when no nucleon with a larger Fermi momentum is left to swap in - the case
/// Geant4 then ignores (header note 2).
__host__ __device__ inline bool reduce_sum(Nucleus3D& nuc, Nucleus3DScratch& sc,
                                           NucleusReport& rep) {
  const int a = nuc.my_a;
  for (;;) {
    Vec3d sum{0.0, 0.0, 0.0};
    const double p_fermi = sc.fermi_p[a - 1];
    for (int i = 0; i < a - 1; ++i) { sum = sum + sc.momentum[i]; }

    if (g4gpu::mag(sum) <= p_fermi) {
      sc.momentum[a - 1] = Vec3d{-sum.x, -sum.y, -sum.z};
      return true;
    }

    const Vec3d test_dir = g4gpu::normalize(sum);
    const int n = a - 1;
    for (int k = 0; k < n; ++k) {
      const Vec3d delta = 2.0 * (g4gpu::dot(sc.momentum[k], test_dir)) * test_dir;
      sc.test_sums[k].vector = delta;
      sc.test_sums[k].size = g4gpu::mag(delta);
      sc.test_sums[k].index = k;
    }
    // std::sort on `Size`; insertion sort here, stable. Exact ties are reachable only when two
    // momenta are both exactly zero (the zeroed-proton case), and then both deltas are the
    // zero vector and the order between them cannot change any later arithmetic.
    for (int i = 1; i < n; ++i) {
      NucleusSortEntry key = sc.test_sums[i];
      int j = i - 1;
      while (j >= 0 && key.size < sc.test_sums[j].size) {
        sc.test_sums[j + 1] = sc.test_sums[j];
        --j;
      }
      sc.test_sums[j + 1] = key;
    }

    int index = n;
    while (g4gpu::mag(sum - sc.test_sums[--index].vector) > p_fermi && index > 0) {
      if (g4gpu::mag(sum) > g4gpu::mag(sum - sc.test_sums[index].vector)) {
        sc.momentum[sc.test_sums[index].index] =
            sc.momentum[sc.test_sums[index].index] - sc.test_sums[index].vector;
        sum = sum - sc.test_sums[index].vector;
      }
    }

    if (g4gpu::mag(sum - sc.test_sums[index].vector) <= p_fermi) {
      int best = -1;
      double p_best = 2.0 * p_fermi;
      const double last_mag = g4gpu::mag(sc.momentum[a - 1]);
      for (int k = 0; k <= index; ++k) {
        const double p_try = g4gpu::mag(sc.test_sums[k].vector - sum);
        if (p_try < p_fermi && std::abs(last_mag - p_try) < p_best) {
          p_best = std::abs(last_mag - p_try);
          best = k;
        }
      }
      if (best < 0) {
        // G4HadronicException("Logic error in ReduceSum()"). A kernel cannot throw.
        rep.reduce_sum_logic = true;
        return false;
      }
      sc.momentum[sc.test_sums[best].index] =
          sc.momentum[sc.test_sums[best].index] - sc.test_sums[best].vector;
      sc.momentum[a - 1] = sc.test_sums[best].vector - sum;
      return true;
    }

    // Find a nucleon with a bigger Fermi momentum than the last one's and swap it in, then
    // start over. Geant4 recurses; this iterates, which is the same thing because the call is
    // in tail position and nothing follows it.
    int swapit = -1;
    while (swapit < a - 1) {
      if (sc.fermi_p[++swapit] > p_fermi) { break; }
    }
    if (swapit == a - 1) { return false; }

    Nucleon tn = nuc.nucleons[swapit];
    nuc.nucleons[swapit] = nuc.nucleons[a - 1];
    nuc.nucleons[a - 1] = tn;
    Vec3d tm = sc.momentum[swapit];
    sc.momentum[swapit] = sc.momentum[a - 1];
    sc.momentum[a - 1] = tm;
    double tf = sc.fermi_p[swapit];
    sc.fermi_p[swapit] = sc.fermi_p[a - 1];
    sc.fermi_p[a - 1] = tf;
  }
}

/// G4Fancy3DNucleus::ChooseFermiMomenta.
///
/// A momentum for every nucleon including the last, "in case we swap nucleons"; then ReduceSum;
/// then the four-momentum is assembled as `(p, PDGMass - BindingEnergy/A)` - deliberately off
/// the mass shell, header note 1.
///
/// The proton branch: the local Fermi momentum is capped so that the Fermi energy stays above
/// the Coulomb barrier, `eMax = sqrt(pF^2 + m^2) - barrier`, and the momentum re-drawn until it
/// fits inside the new cap. The ALREADY-DRAWN momentum is tested first and kept if it fits, so
/// the number of deviates consumed depends on the first draw. When `eMax <= m` - a place where
/// the local Fermi energy cannot clear the barrier, which `ChoosePositions` should have already
/// rejected for a proton - Geant4 emits a JustWarning and sets the momentum to exactly zero;
/// that is reproduced, and counted in `proton_momentum_zeroed`.
///
/// The commented-out line under the four-momentum assignment - `SetBindingEnergy(0.5 pF^2/m)`,
/// dated "GF 11-05-2011" - is the non-relativistic kinetic energy the author considered putting
/// in place of the uniform `BindingEnergy()/A` that `Init` writes a moment later. It is not
/// live and is recorded rather than ported.
template <typename Rng>
__host__ __device__ inline void choose_fermi_momenta(Nucleus3D& nuc, Nucleus3DScratch& sc,
                                                     NucleusReport& rep, Rng& rng) {
  for (int i = 0; i < nuc.my_a; ++i) {
    sc.momentum[i] = Vec3d{0.0, 0.0, 0.0};
    sc.fermi_p[i] = 0.0;
  }

  // `for (G4int ntry=0; ntry<1; ntry++)` - one iteration. Header note 2.
  for (int ntry = 0; ntry < 1; ++ntry) {
    for (int i = 0; i < nuc.my_a; ++i) {
      const double dens = nuc.density.density(nuc.nucleons[i].position);
      sc.fermi_p[i] = nuc.fermi.fermi_momentum(dens);
      Vec3d mom = nuc.fermi.momentum(dens, rng);
      if (nuc.nucleons[i].type == kProton) {
        const double m = nucleon_pdg_mass(kProton);
        const double e_max =
            std::sqrt(sc.fermi_p[i] * sc.fermi_p[i] + m * m) - nuc.coulomb_barrier();
        if (e_max > m) {
          const double pmax2 = e_max * e_max - m * m;
          sc.fermi_p[i] = std::sqrt(pmax2);
          while (g4gpu::mag2(mom) > pmax2) {
            mom = nuc.fermi.momentum(dens, rng, sc.fermi_p[i]);
          }
        } else {
          ++rep.proton_momentum_zeroed;
          mom = Vec3d{0.0, 0.0, 0.0};
        }
      }
      sc.momentum[i] = mom;
    }
    if (reduce_sum(nuc, sc, rep)) { break; }
    rep.reduce_sum_failed = true;
  }

  for (int i = 0; i < nuc.my_a; ++i) {
    const double energy = nuc.nucleons[i].pdg_mass() -
                          nuc.binding_energy() / static_cast<double>(nuc.my_a);
    nuc.nucleons[i].momentum = LorentzVector(sc.momentum[i], energy);
  }
}

/// G4Fancy3DNucleus::Init(A, Z, numberOfLambdas).
///
/// The order is the whole content of the method and none of it is interchangeable:
/// `myL = max(L, 0)`; the density is chosen by `myA < 17` (shell model below, Fermi at and
/// above) with the hard-core distance raised to 0.9 fm for A = 12 alone; `theFermi.Init`;
/// `ChooseNucleons` (which needs nothing but A and Z); `ChoosePositions` (which needs the
/// density and, for protons, the Coulomb barrier); `CenterNucleons` for A = 12; then
/// `ChooseFermiMomenta` (which needs the positions); and finally one binding energy per
/// nucleon, `GetBindingEnergy(A, Z)/A`, written onto every nucleon including the Lambdas -
/// "we neglect eventual Lambdas as far as the density of nuclear levels and the Fermi level
/// are concerned", the source's own comment.
///
/// The `NON_INTEGER_A_Z` overload, `#if defined`-ed out in 11.1.1, would have rounded a
/// fractional A as `(U() > A - floor(A)) ? floor(A) : floor(A)+1` - note the inverted sense,
/// which makes a fractional part of 0.25 round UP three times in four. Not ported; recorded.
template <typename Rng>
__host__ __device__ inline NucleusReport nucleus_init(Nucleus3D& nuc, Nucleus3DScratch& sc,
                                                     int the_a, int the_z, Rng& rng,
                                                     int number_of_lambdas = 0) {
  NucleusReport rep;
  if (the_a > nuc.capacity || the_a > sc.capacity || the_a < 1) {
    rep.capacity = true;
    rep.refused_a = the_a;
    rep.refused_z = the_z;
    return rep;
  }

  nuc.current = -1;
  nuc.nucleondistance = 0.8 * deex::fermi();
  nuc.my_z = the_z;
  nuc.my_a = the_a;
  nuc.my_l = (number_of_lambdas > 0) ? number_of_lambdas : 0;
  nuc.excitation = 0.0;
  for (int i = 0; i < the_a; ++i) { nuc.nucleons[i] = Nucleon{}; }

  if (nuc.my_a < 17) {
    nuc.density = make_shell_model_density(nuc.my_a, nuc.my_z);
    if (nuc.my_a == 12) { nuc.nucleondistance = 0.9 * deex::fermi(); }
  } else {
    nuc.density = make_fermi_density(nuc.my_a, nuc.my_z);
  }

  nuc.fermi.init(nuc.my_a, nuc.my_z);

  choose_nucleons(nuc, rng);

  if (nuc.my_a != 12) {
    if (!choose_positions(nuc, sc, rng)) {
      rep.placement_failed = true;
      rep.refused_a = the_a;
      rep.refused_z = the_z;
      return rep;
    }
  } else {
    if (!choose_positions_c12(nuc, sc, rng)) {
      rep.cluster_placement_failed = true;
      rep.refused_a = the_a;
      rep.refused_z = the_z;
      return rep;
    }
  }

  // "This would introduce a bias" is the comment Geant4 puts on this line, and the line runs.
  // What it does is remove the offset the 0.866 triangle's centroid leaves behind; what the
  // comment warns about is that recentring a sampled configuration correlates its nucleons.
  if (nuc.my_a == 12) { nuc.center_nucleons(); }

  choose_fermi_momenta(nuc, sc, rep, rng);

  const double e_binding = nuc.binding_energy() / static_cast<double>(nuc.my_a);
  for (int i = 0; i < nuc.my_a; ++i) { nuc.nucleons[i].binding_energy = e_binding; }

  return rep;
}

}  // namespace g4gpu::bic

#endif
