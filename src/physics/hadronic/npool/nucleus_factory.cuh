// P38: THE NUCLEUS FACTORY - fresh G4Fancy3DNucleus samples in bulk, one lane per nucleon.
//
// `bic::nucleus_init` (bic/nucleus/fancy_3d_nucleus.cuh, G4Fancy3DNucleus::Init) builds one
// nucleus on one thread: ChooseNucleons, ChoosePositions (or the C12 cluster branch),
// CenterNucleons for A = 12, ChooseFermiMomenta with ReduceSum, and one binding energy per
// nucleon. On the transport's one-thread interaction blocks that is 1.6 ms of thread time and
// 1,517 uniforms an O16 (docs/RISK.md V297, V300) - 95% of a Binary cascade's draws and 99% of a
// light-ion reaction's. This file samples the SAME DISTRIBUTION with a block per nucleus and a
// lane per nucleon; every formula is the thread path's own function, called per nucleon.
//
// Under the user's rule of 2026-11-19 (docs/RISK.md V302) the draws inside an interaction may be
// consumed in another order, so the factory is held to the distribution and not to the stream:
// `tests/test_npool_nucleus.cu` compares it with `nucleus_init` nuclide by nuclide (V319).
//
// ## WHAT EACH PHASE IS, AND WHY IT IS THE SAME DISTRIBUTION
//
// THE GENERATOR IS KEYED PER (nucleus, lane, phase) - `factory_stream` - never per thread or
// per launch, so a nucleus is the same to the bit whatever block, grid or occupancy ran it.
//
// 1. ChooseNucleons. Geant4 draws one uniform per loop turn; while both kinds remain, the turn
//    assigns the next slot a proton with probability Z/A, and once one kind is exhausted every
//    later turn either skips or assigns the other kind. So the slot types are an i.i.d.
//    Bernoulli(Z/A) sequence b_1, b_2, ... up to the slot T at which one count reaches its total,
//    and the other kind after it. Here lane k draws b_k (A draws always suffice, T <= A) and
//    lane 0 walks the A values once to find T. Same law, A draws where Geant4's averages
//    A + Z ln(A/(A-Z)).
//
// 2. ChoosePositions (A != 12). A trial is a point uniform in the ball of radius
//    `GetNuclearRadius(0.001)`, kept with probability `GetRelativeDensity`, and for a proton
//    refused where the local Fermi energy is at or under the Coulomb barrier; then the hard
//    core: refused if within `nucleondistance` of any nucleon already placed. Every test but
//    the hard core is independent of the other nucleons, so lane i draws its OWN candidate
//    stream already passed through them. The hard core is a WAVEFRONT:
//
//      * every round each lane that is not final tests its held candidate against the current
//        candidates of every lower lane;
//      * the final prefix grows by the run of conflict-free lanes after it (lane 0 decides);
//      * a lane whose candidate conflicts with a lane that IS final throws it away and draws
//        its next one - that test is against a final position, so its outcome is the serial
//        loop's; a conflict with a lane that is not yet final decides nothing, and the
//        candidate is HELD, not redrawn, until it can be tested.
//
//    So lane i's position is the first candidate of its own stream that clears the FINAL
//    positions of lanes 0..i-1, which is the serial loop's definition with lane i's i.i.d.
//    stream in place of the serial stream after nucleon i-1 was placed - the same acceptance
//    process in the same order, with independent draws, and so the same joint law. Geant4's
//    attempt budget (1000 A trials for the whole nucleus, `mod_util001`) is the sum of every
//    lane's trials, which has the serial total's law; a nucleus over it is refused as
//    `placement_failed`, as the thread path refuses it.
//
//    The `G4RandFlat::shootArray` block of `min(600, 9(A-i))` uniforms consumed from the top
//    down is a property of Geant4's STREAM: its values are i.i.d. uniforms whichever order they
//    are read in, and the leftovers it discards are never read. A lane draws the three
//    coordinates directly, which is why a factory nucleus costs ~4 uniforms a trial where the
//    thread path's averages 1,517 an O16.
//
// 3. ChoosePositions (A == 12): three alpha clusters on a triangle (bic::choose_positions_c12).
//    Nucleon 0 is placed with no test; nucleons 1-11 test against every lower nucleon, which is
//    the same wavefront with Gaussian candidates around their cluster's corner; each cluster's
//    budget of 10,000 refusals is counted across its lanes. The random rotation's two draws
//    are lane 0's; `CenterNucleons` is called on lane 0 exactly as Init calls it.
//
// 4. ChooseFermiMomenta. Each nucleon's momentum depends only on its own position (the local
//    density) and its kind, so it is per lane; `ReduceSum` - a sort and a scan over the whole
//    nucleus that may swap nucleons - is `bic::reduce_sum` on lane 0, unchanged, over the
//    factory's arrays, followed per lane by the four-momentum and binding energy Init writes.
//
// ## REFUSED, by name
//
//   * hyper-nuclei (`my_l > 0`): the thread path refuses their MASS, and no caller asks the
//     factory for one; `FactoryJob::a` above `kFactoryMaxA` (256) is refused as `capacity`.
#ifndef G4GPU_NPOOL_NUCLEUS_FACTORY_CUH
#define G4GPU_NPOOL_NUCLEUS_FACTORY_CUH

#include <cmath>

#include "core/rng.cuh"
#include "physics/hadronic/bic/nucleus/fancy_3d_nucleus.cuh"

namespace g4gpu::npool {

/// The widest nucleus a factory block takes: `bic::kSlotNucleons`, which Pb208 and U238 fit.
inline constexpr int kFactoryMaxA = 256;

/// The phase a lane's draws belong to - the third word of its key.
enum FactoryPhase : unsigned {
  kFactoryTypes = 1u,
  kFactoryPositions = 2u,
  kFactoryMomenta = 3u,
  kFactoryRotation = 4u,
};

/// One lane's stream in one phase: Philox with (key_lo, key_hi << 8 | lane, P38 salt | tag << 4
/// | phase). `key_hi` must stay below 2^24 so that no two (key_hi, lane) pairs share a word, and
/// `tag` below 2^12. P39 (docs/RISK.md V321) adds the tag - which nucleus of a cascade this is,
/// and whether it is the projectile or the target - so that a cascade keyed (key_lo, key_hi)
/// gets a different nucleus for every try; tag 0 is P38's stream, word for word.
__host__ __device__ inline Philox<double> factory_stream(unsigned key_lo, unsigned key_hi,
                                                        int lane, unsigned phase,
                                                        unsigned tag = 0u) {
  return Philox<double>(key_lo, (key_hi << 8) | static_cast<unsigned>(lane & 0xff),
                        0x50380000u | ((tag & 0xfffu) << 4) | phase);
}

/// The largest tag `factory_stream` keeps apart.
inline constexpr unsigned kFactoryMaxTag = 0xfffu;

/// What one launch samples: `n` nuclei of one nuclide, nucleus k keyed (key_lo + k, key_hi).
struct FactoryJob {
  int a = 0;
  int z = 0;
  int n = 0;
  unsigned key_lo = 0;
  unsigned key_hi = 0;
  /// TEST-ONLY perturbation of the hard-core distance (`tests/test_npool_nucleus.cu`'s
  /// anti-vacuity run halves it). 1.0 is G4Fancy3DNucleus's distance and every caller's.
  double distance_scale = 1.0;
};

/// Where the nuclei go. Nucleus k is `nucleons[k*a .. k*a+a-1]`, the `bic::Nucleon` layout the
/// thread path's `Nucleus3D` points at, so a factory nucleus can be handed to either path.
/// `momentum`, `fermi_p` and `test_sums` are ReduceSum's scratch, `a` entries a nucleus.
struct FactoryOutput {
  bic::Nucleon* nucleons = nullptr;
  bic::NucleusReport* reports = nullptr;
  deex::Vec3d* momentum = nullptr;
  double* fermi_p = nullptr;
  bic::NucleusSortEntry* test_sums = nullptr;
  int* rounds = nullptr;   ///< optional: the wavefront's rounds per nucleus, for V319
};

/// The per-nucleus constants `G4Fancy3DNucleus::Init` sets before it samples anything - the
/// first statements of `bic::nucleus_init`, which samples and so cannot be called for them.
/// Also what a caller of the factory builds its `Nucleus3D` view with.
__host__ __device__ inline void factory_nucleus_header(bic::Nucleus3D& nuc,
                                                       bic::Nucleon* nucleons, int a, int z) {
  nuc.nucleons = nucleons;
  nuc.capacity = a;
  nuc.current = -1;
  nuc.nucleondistance = 0.8 * deex::fermi();
  nuc.my_z = z;
  nuc.my_a = a;
  nuc.my_l = 0;
  nuc.excitation = 0.0;
  if (a < 17) {
    nuc.density = bic::make_shell_model_density(a, z);
    if (a == 12) { nuc.nucleondistance = 0.9 * deex::fermi(); }
  } else {
    nuc.density = bic::make_fermi_density(a, z);
  }
  nuc.fermi.init(a, z);
}

/// The block's shared state for one nucleus. `kW` is the block width, a multiple of 32 >= A.
template <int kW>
struct FactoryShared {
  deex::Vec3d pos[kW];
  int type_draw[kW];
  int type[kW];
  unsigned char fin[kW];
  unsigned char conf[kW];
  int frontier;
  int done;
  int fail;
  int trials;
  int zeroed;
  int cluster_refusals[3];
  int rounds;
  double rot[4];   ///< cos phi, sin phi, cos theta, sin theta of the C12 rotation
};

/// One candidate of lane `lane`'s stream for the A != 12 branch: the trial loop of
/// `bic::choose_positions` up to (not including) the hard core, repeated until a trial passes.
/// Returns false when the lane alone has used the whole nucleus's budget.
template <typename Rng>
__device__ inline bool factory_candidate(const bic::Nucleus3D& nuc, int type, double max_r,
                                         double barrier, int budget, Rng& rng, int& trials,
                                         deex::Vec3d& out) {
  for (;;) {
    if (trials >= budget) { return false; }
    deex::Vec3d p{0.0, 0.0, 0.0};
    do {
      const double x = 2.0 * rng.uniform() - 1.0;
      const double y = 2.0 * rng.uniform() - 1.0;
      const double zc = 2.0 * rng.uniform() - 1.0;
      p = deex::Vec3d{x, y, zc};
    } while (g4gpu::mag2(p) > 1.0);
    p = p * max_r;
    ++trials;
    if (!(rng.uniform() < nuc.density.relative_density(p))) { continue; }
    if (type == bic::kProton) {
      const double p_fermi = nuc.fermi.fermi_momentum(nuc.density.density(p));
      const double m = bic::nucleon_pdg_mass(bic::kProton);
      const double e_fermi = std::sqrt(p_fermi * p_fermi + m * m) - m;
      if (e_fermi <= barrier) { continue; }
    }
    out = p;
    return true;
  }
}

/// The C12 branch's candidate for nucleon `lane` (1..11): `bic::gauss_triplet` about its
/// cluster's corner, as `bic::choose_positions_c12` draws it.
template <typename Rng>
__device__ inline deex::Vec3d factory_c12_candidate(int lane, Rng& rng) {
  const double l_base = 3.05 * deex::fermi();
  const double disp = 0.552;  // `bic::choose_positions_c12`'s; a variance used as a sigma
  const deex::Vec3d corner[3] = {deex::Vec3d{l_base / 2.0, 0.0, 0.0},
                                 deex::Vec3d{-l_base / 2.0, 0.0, 0.0},
                                 deex::Vec3d{0.0, l_base * 0.866, 0.0}};
  return bic::gauss_triplet(rng, disp) * deex::fermi() + corner[lane / 4];
}

/// ONE nucleus as the block builds it: its nuclide, its key and where its nucleons, report and
/// ReduceSum scratch go. `factory_build_one` (a job's k-th nucleus) and the request list of the
/// flat cascade (P39, `nucleus_factory_list_kernel`) both come down to this.
struct FactoryNucleus {
  int a = 0;
  int z = 0;
  unsigned key_lo = 0;
  unsigned key_hi = 0;
  unsigned tag = 0;
  double distance_scale = 1.0;
  bic::Nucleon* nucleons = nullptr;   ///< [a]
  bic::NucleusReport* report = nullptr;
  int* rounds = nullptr;              ///< optional
  deex::Vec3d* momentum = nullptr;    ///< [a] ReduceSum's scratch
  double* fermi_p = nullptr;          ///< [a]
  bic::NucleusSortEntry* test_sums = nullptr;  ///< [a]
};

/// One nucleus on one block. Every lane of the block calls it; lanes >= A only meet the
/// barriers.
template <int kW>
__device__ inline void factory_build(const FactoryNucleus& job, FactoryShared<kW>& sh) {
  const int lane = static_cast<int>(threadIdx.x);
  const int a = job.a;
  const int z = job.z;
  const bool live = lane < a;
  const unsigned key_lo = job.key_lo;
  const unsigned tag = job.tag;
  bic::Nucleon* nucleons = job.nucleons;

  bic::Nucleus3D nuc;
  factory_nucleus_header(nuc, nucleons, a, z);
  const double nd = nuc.nucleondistance * job.distance_scale;
  const double nd2 = nd * nd;
  bic::NucleusReport rep;

  // ---- 1. ChooseNucleons
  if (live) {
    Philox<double> rt = factory_stream(key_lo, job.key_hi, lane, kFactoryTypes, tag);
    const double prob_proton = static_cast<double>(z) / static_cast<double>(a);
    sh.type_draw[lane] = (rt.uniform() < prob_proton) ? bic::kProton : bic::kNeutron;
  }
  if (lane == 0) {
    sh.frontier = 0;
    sh.done = 0;
    sh.fail = 0;
    sh.trials = 0;
    sh.zeroed = 0;
    sh.rounds = 0;
    sh.cluster_refusals[0] = sh.cluster_refusals[1] = sh.cluster_refusals[2] = 0;
  }
  __syncthreads();
  if (lane == 0) {
    int protons = 0, neutrons = 0, other = 0;
    for (int i = 0; i < a; ++i) {
      if (other != 0) {
        sh.type[i] = other;
        continue;
      }
      const int t = sh.type_draw[i];
      sh.type[i] = t;
      if (t == bic::kProton) { ++protons; } else { ++neutrons; }
      if (protons == z) {
        other = bic::kNeutron;
      } else if (neutrons == a - z) {
        other = bic::kProton;
      }
    }
  }
  if (lane < kW) { sh.fin[lane] = live ? 0 : 1; }
  __syncthreads();
  const int my_type = live ? sh.type[lane] : bic::kNucleonNone;

  // ---- 2. ChoosePositions, the wavefront
  Philox<double> rp = factory_stream(key_lo, job.key_hi, lane, kFactoryPositions, tag);
  const bool c12 = (a == 12);
  const double max_r = nuc.nuclear_radius(0.001);
  const double barrier = nuc.coulomb_barrier();
  const int budget = 1000 * a;
  int trials = 0;
  bool have = false;
  deex::Vec3d cand{0.0, 0.0, 0.0};
  if (c12 && lane == 0) {
    // "The FIRST nucleon of the first cluster: no overlap test."
    sh.pos[0] = bic::gauss_triplet(rp, 0.552) * deex::fermi() +
                deex::Vec3d{3.05 * deex::fermi() / 2.0, 0.0, 0.0};
    sh.fin[0] = 1;
    sh.frontier = 1;
  }
  __syncthreads();
  for (int guard = 0; guard < 1000000; ++guard) {
    if (live && !sh.fin[lane] && !have) {
      if (c12) {
        cand = factory_c12_candidate(lane, rp);
        have = true;
      } else {
        have = factory_candidate(nuc, my_type, max_r, barrier, budget, rp, trials, cand);
        if (!have) { sh.fail = 1; }
      }
    }
    if (live && !sh.fin[lane]) { sh.pos[lane] = cand; }
    __syncthreads();
    if (sh.fail) { break; }
    bool conflict = false;
    if (live && !sh.fin[lane]) {
      for (int j = 0; j < lane; ++j) {
        if (g4gpu::mag2(sh.pos[lane] - sh.pos[j]) <= nd2) {
          conflict = true;
          break;
        }
      }
    }
    sh.conf[lane] = conflict ? 1 : 0;
    __syncthreads();
    if (lane == 0) {
      int i = sh.frontier;
      while (i < a && !sh.conf[i]) {
        sh.fin[i] = 1;
        ++i;
      }
      sh.frontier = i;
      sh.done = (i >= a) ? 1 : 0;
      ++sh.rounds;
    }
    __syncthreads();
    if (sh.done) { break; }
    // A conflict with a FINAL lane is the serial loop's verdict on this candidate.
    if (live && !sh.fin[lane] && conflict) {
      for (int j = 0; j < lane; ++j) {
        if (sh.fin[j] && g4gpu::mag2(sh.pos[lane] - sh.pos[j]) <= nd2) {
          have = false;
          if (c12) { atomicAdd(&sh.cluster_refusals[lane / 4], 1); }
          break;
        }
      }
    }
    __syncthreads();
  }
  if (!c12 && live) { atomicAdd(&sh.trials, trials); }
  __syncthreads();
  if (lane == 0) {
    if (c12) {
      for (int c = 0; c < 3; ++c) {
        if (sh.cluster_refusals[c] >= 10000) { sh.fail = 1; }
      }
      if (sh.fail) { rep.cluster_placement_failed = true; }
    } else if (sh.fail || sh.trials >= budget) {
      sh.fail = 1;
      rep.placement_failed = true;
    }
    if (sh.fail) {
      rep.refused_a = a;
      rep.refused_z = z;
    }
  }
  __syncthreads();
  if (sh.fail) {
    if (lane == 0) {
      *job.report = rep;
      if (job.rounds != nullptr) { *job.rounds = sh.rounds; }
    }
    return;
  }

  if (live) {
    bic::Nucleon nn;
    nn.position = sh.pos[lane];
    nn.type = my_type;
    nucleons[lane] = nn;
  }
  __syncthreads();
  if (c12) {
    // The random rotation (`rotateZ(2 pi U)`, `rotateY(acos(2U - 1))`), then CenterNucleons -
    // both as `bic::choose_positions_c12` and `bic::nucleus_init` write them, on lane 0.
    if (lane == 0) {
      Philox<double> rr = factory_stream(key_lo, job.key_hi, 0, kFactoryRotation, tag);
      const double phi = g4gpu::units::twopi<double>() * rr.uniform();
      const double theta = std::acos(2.0 * rr.uniform() - 1.0);
      sh.rot[0] = std::cos(phi);
      sh.rot[1] = std::sin(phi);
      sh.rot[2] = std::cos(theta);
      sh.rot[3] = std::sin(theta);
    }
    __syncthreads();
    if (live) {
      const deex::Vec3d p = nucleons[lane].position;
      const double cp = sh.rot[0], sp = sh.rot[1], ct = sh.rot[2], st = sh.rot[3];
      const double x1 = p.x * cp - p.y * sp;
      const double y1 = p.x * sp + p.y * cp;
      const double z1 = p.z;
      nucleons[lane].position = deex::Vec3d{x1 * ct + z1 * st, y1, -x1 * st + z1 * ct};
    }
    __syncthreads();
    if (lane == 0) { nuc.center_nucleons(); }
    __syncthreads();
  }

  // ---- 4. ChooseFermiMomenta: per nucleon, then ReduceSum on lane 0
  deex::Vec3d* momentum = job.momentum;
  double* fermi_p = job.fermi_p;
  if (live) {
    Philox<double> rm = factory_stream(key_lo, job.key_hi, lane, kFactoryMomenta, tag);
    const deex::Vec3d pos = nucleons[lane].position;
    const double dens = nuc.density.density(pos);
    double pf = nuc.fermi.fermi_momentum(dens);
    deex::Vec3d mom = nuc.fermi.momentum(dens, rm);
    if (my_type == bic::kProton) {
      const double m = bic::nucleon_pdg_mass(bic::kProton);
      const double e_max = std::sqrt(pf * pf + m * m) - nuc.coulomb_barrier();
      if (e_max > m) {
        const double pmax2 = e_max * e_max - m * m;
        pf = std::sqrt(pmax2);
        while (g4gpu::mag2(mom) > pmax2) { mom = nuc.fermi.momentum(dens, rm, pf); }
      } else {
        atomicAdd(&sh.zeroed, 1);
        mom = deex::Vec3d{0.0, 0.0, 0.0};
      }
    }
    fermi_p[lane] = pf;
    momentum[lane] = mom;
  }
  __syncthreads();
  if (lane == 0) {
    rep.proton_momentum_zeroed = sh.zeroed;
    bic::Nucleus3DScratch sc;
    sc.momentum = momentum;
    sc.fermi_p = fermi_p;
    sc.test_sums = job.test_sums;
    sc.capacity = a;
    if (!bic::reduce_sum(nuc, sc, rep)) { rep.reduce_sum_failed = true; }
    *job.report = rep;
    if (job.rounds != nullptr) { *job.rounds = sh.rounds; }
  }
  __syncthreads();
  if (live) {
    const double e_binding = nuc.binding_energy() / static_cast<double>(a);
    bic::Nucleon& nn = nucleons[lane];
    nn.momentum = bic::LorentzVector(momentum[lane], nn.pdg_mass() - e_binding);
    nn.binding_energy = e_binding;
  }
}

/// A job's nucleus k: keyed (key_lo + k, key_hi), tag 0, its arrays at k * a.
template <int kW>
__device__ inline void factory_build_one(const FactoryJob& job, const FactoryOutput& out, int k,
                                         FactoryShared<kW>& sh) {
  const size_t off = static_cast<size_t>(k) * job.a;
  FactoryNucleus n;
  n.a = job.a;
  n.z = job.z;
  n.key_lo = job.key_lo + static_cast<unsigned>(k);
  n.key_hi = job.key_hi;
  n.tag = 0u;
  n.distance_scale = job.distance_scale;
  n.nucleons = out.nucleons + off;
  n.report = out.reports + k;
  n.rounds = (out.rounds != nullptr) ? out.rounds + k : nullptr;
  n.momentum = out.momentum + off;
  n.fermi_p = out.fermi_p + off;
  n.test_sums = out.test_sums + off;
  factory_build<kW>(n, sh);
}

/// The factory: one block of `kW` lanes per nucleus, the grid striding over the job's nuclei.
template <int kW>
__global__ void __launch_bounds__(kW) nucleus_factory_kernel(FactoryJob job, FactoryOutput out) {
  __shared__ FactoryShared<kW> sh;
  for (int k = static_cast<int>(blockIdx.x); k < job.n; k += static_cast<int>(gridDim.x)) {
    factory_build_one<kW>(job, out, k, sh);
    __syncthreads();
  }
}

/// The block width for a nuclide: the smallest of 32, 64, 128, 256 that holds A.
__host__ __device__ inline int factory_width(int a) {
  return (a <= 32) ? 32 : (a <= 64) ? 64 : (a <= 128) ? 128 : 256;
}

// =============================================================================================
// P39: THE REQUEST LIST - nuclei of MIXED nuclides, each written where its cascade reads it.
//
// The flat cascade (bic/flat/, docs/RISK.md V321) asks for one nucleus a set-up (the Binary
// target; the light-ion reaction's projectile and target), each of its own (A, Z) and key, into
// the cascade's own record. A request names all of that; the list kernel builds request k on
// block k (striding), the count read on the device, so the host launches without knowing it.
// ReduceSum's scratch is the block's shared memory, not a global buffer: nothing is allocated
// per nucleus beyond the request itself, and the nucleons land in the record.
// =============================================================================================

/// One nucleus a cascade asked for.
struct FactoryRequest {
  bic::Nucleon* nucleons = nullptr;      ///< the record's array, at least `a` long
  bic::NucleusReport* report = nullptr;  ///< the record's report
  int a = 0;
  int z = 0;
  unsigned key_lo = 0;
  unsigned key_hi = 0;
  unsigned tag = 0;
};

/// Where a set-up writes its requests: two lists by block width, `count[0]` and `count[1]`
/// their lengths. Each list holds `cap` requests - every lane's worst case (one a Binary
/// cascade, two a light-ion reaction) - so a push never finds a list full: if it could, which
/// cascade fell back to its lane would depend on the atomic order, not on the cascade.
struct FactoryLists {
  FactoryRequest* req32 = nullptr;
  FactoryRequest* req64 = nullptr;
  int* count = nullptr;
  int cap = 0;
};

/// Appends a request; false (the caller samples on its lane) if A is outside 1..64 or the list
/// is full, which the sizing above makes unreachable.
__device__ inline bool factory_push(const FactoryLists& lists, const FactoryRequest& r) {
  if (r.a < 1 || r.a > 64 || r.tag > kFactoryMaxTag) { return false; }
  const int w = (r.a <= 32) ? 0 : 1;
  const int k = atomicAdd(&lists.count[w], 1);
  if (k >= lists.cap) { return false; }
  (w == 0 ? lists.req32 : lists.req64)[k] = r;
  return true;
}

/// `min(count[0], cap)` requests of width `kW` (every A <= kW), one block of `kW` lanes each.
template <int kW>
__global__ void __launch_bounds__(kW) nucleus_factory_list_kernel(const FactoryRequest* req,
                                                                  const int* count, int cap) {
  __shared__ FactoryShared<kW> sh;
  // `NucleusSortEntry` has default member initialisers, which a `__shared__` object may not
  // run, so ReduceSum's three arrays are raw shared storage of the right size and alignment.
  __shared__ __align__(16) unsigned char raw_mom[kW * sizeof(deex::Vec3d)];
  __shared__ __align__(16) unsigned char raw_fp[kW * sizeof(double)];
  __shared__ __align__(16) unsigned char raw_ts[kW * sizeof(bic::NucleusSortEntry)];
  const int n = (*count < cap) ? *count : cap;
  for (int k = static_cast<int>(blockIdx.x); k < n; k += static_cast<int>(gridDim.x)) {
    const FactoryRequest r = req[k];
    FactoryNucleus job;
    job.a = r.a;
    job.z = r.z;
    job.key_lo = r.key_lo;
    job.key_hi = r.key_hi;
    job.tag = r.tag;
    job.nucleons = r.nucleons;
    job.report = r.report;
    job.momentum = reinterpret_cast<deex::Vec3d*>(raw_mom);
    job.fermi_p = reinterpret_cast<double*>(raw_fp);
    job.test_sums = reinterpret_cast<bic::NucleusSortEntry*>(raw_ts);
    factory_build<kW>(job, sh);
    __syncthreads();
  }
}

}  // namespace g4gpu::npool

#endif
