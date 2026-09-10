// Fermi break-up: the light-fragment channel that runs INSTEAD of evaporation for Z < 9 and
// A < 17, and the pool of pre-computed two-body channels it walks.
//
// Transcribed from G4FermiBreakUpVI, G4FermiFragmentsPoolVI, G4FermiFragment, G4FermiPair,
// G4FermiChannels and G4FermiDecayProbability (11.1.1).
//
// The model is a cascade of TWO-BODY decays, not an n-body phase-space decay. G4FermiBreakUpVI
// picks one pair of pool fragments, boosts them apart, and pushes both onto a work list that
// the loop then re-enters; a fragment that has no channel left is a final product. So
// G4FermiPhaseSpaceDecay - the class in the same directory that samples an n-body final state -
// is **not used by this model at all**; only G4BinaryCascade instantiates it. It is not ported
// and its absence is not a gap in this package.
//
// Everything expensive is done once, at initialisation, by the pool:
//
//   991 fragments   the eight hard-coded stable particles (n, p, d, t, He3, alpha AND the
//                   unbound He5 and Li5), then every level below 20 MeV of every nuclide with
//                   Z < 9 and A < 17 that has level data
//   the pairs       every ordered pair whose combined (Z, A) is inside the window and whose
//                   combined mass plus Coulomb barrier is at or below the total energy of some
//                   pool fragment with that (Z, A)
//   the channels    per pool fragment, the list of pairs it can decay into, with the
//                   cumulative decay probability already normalised
//
// The pool build is host-side for the same reason the level table's is: it reads the level
// data. The result is flat arrays a kernel indexes, and nothing in the decay path allocates.
//
// **The pool's excitation energies are float-truncated.** G4FermiFragmentsPoolVI::Initialise
// reads `G4float exc = man->LevelEnergy(i)` and then widens it back to double, so a pool
// fragment's excitation is the level energy rounded to 24 bits - and it is that rounded value
// that ClosestChannels compares an incoming excitation against, within a 10 eV tolerance. The
// truncation is larger than the tolerance above about 80 MeV of level energy, which the 20 MeV
// limit keeps out of reach; below it the two are the same number and the float still has to be
// there, because IsInThePool's de-duplication and the `exc >= elimf` cut are both done on it.
//
// One quirk reproduced because it decides which probabilities are used:
// G4FermiBreakUpVI::BreakFragment sets `excitation` from the INITIAL fragment and never
// updates it, while the loop over the work list re-uses SampleDecay for every secondary. So
// `|excitation - chan->GetExcitation()| < 1 MeV` - the test that chooses between the pool's
// pre-normalised static probabilities and a fresh recomputation - is answered for every
// secondary with the primary's excitation. It is not a cache-invalidation bug that shows up as
// a wrong answer; it is a choice between two correct-looking answers, made on stale data.
#ifndef G4GPU_DEEX_FERMI_BREAKUP_CUH
#define G4GPU_DEEX_FERMI_BREAKUP_CUH

#include <cmath>
#include <vector>

#include "core/units.cuh"
#include "data/level_data.cuh"
#include "physics/hadronic/deexcitation/coulomb_barrier.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

/// The window, from the file-scope constants of G4FermiFragmentsPoolVI.hh. They are `static
/// const G4int` at namespace scope in a header, so every translation unit that includes it
/// gets its own copy; the values are 9 and 17 and G4FermiBreakUpVI repeats them in its own
/// constructor as maxZ = 9, maxA = 17.
constexpr int kFermiMaxZ = 9;
constexpr int kFermiMaxA = 17;

/// The largest work list one break-up can produce. G4FermiBreakUpVI reserves 10 and caps the
/// loop at i == 100, so no more than 102 entries are ever read.
constexpr int kFermiMaxWork = 104;

/// G4FermiFragment.
struct FermiFragment {
  int a, z;
  int spin;              ///< 2J, as G4LevelManager::SpinTwo returns it; -1 means unknown
  double excitation;     ///< MeV, float-truncated - see the file header
  double fragment_mass;  ///< G4NucleiProperties::GetNuclearMass(A, Z)
  __host__ __device__ double total_energy() const { return fragment_mass + excitation; }
};

/// G4FermiPair - two indices into the fragment pool. Geant4 stores pointers and compares them
/// by identity in IsInPhysPairs; indices are the same comparison without the pointer.
struct FermiPair {
  int f1, f2;
};

/// The flat pool. `chan_*` is one channel set per (A, j) pool fragment, in the order
/// G4FermiFragmentsPoolVI builds list_f[A] and list_c[A] - which is fragment_pool order
/// restricted to that A, so list_f[A][j] and list_c[A][j] are the same fragment.
struct FermiPool {
  const FermiFragment* frag = nullptr;
  int n_frag = 0;
  /// list_f[A]: `by_a_offset[A]` .. `by_a_offset[A] + by_a_count[A]` index into `by_a`, whose
  /// entries are indices into `frag`.
  const int* by_a = nullptr;
  const int* by_a_offset = nullptr;
  const int* by_a_count = nullptr;
  /// For the k-th entry of `by_a`, the channel set: `ch_offset[k]` .. + `ch_count[k]` index
  /// into `pair_of_channel` and `cum_prob`.
  const int* ch_offset = nullptr;
  const int* ch_count = nullptr;
  const FermiPair* pairs = nullptr;
  const int* pair_of_channel = nullptr;
  const double* cum_prob = nullptr;
  int n_channels = 0;
  int n_pairs = 0;
};

/// G4FermiDecayProbability::ComputeProbability.
///
/// `A` is the DECAYING fragment's mass number and enters as a plain factor; `spin` is its 2J,
/// and a negative spin switches the spin factor off entirely - which is how BreakFragment's
/// first decay differs from the pool's static probabilities, since it passes -1.
///
/// The permutation factor is 0.5 when the two fragments are the SAME POOL ENTRY, not merely
/// the same nuclide: two different levels of the same nuclide are two entries and get 1.0.
__host__ __device__ inline double fermi_decay_probability(const FermiPool& pool, int A, int spin,
                                                           double etot, int i1, int i2) {
  const FermiFragment& f1 = pool.frag[i1];
  const FermiFragment& f2 = pool.frag[i2];
  const double mass1 = f1.total_energy();
  const double mass2 = f2.total_energy();
  const double b_coulomb = deex::coulomb_barrier(f1.a, f1.z, f2.a, f2.z, 0.0);
  if (etot <= mass1 + mass2 + b_coulomb) { return 0.0; }

  const double ekin = etot - mass1 - mass2;
  const double x = mass1 * mass2 / (mass1 + mass2);
  const double mass_factor = x * std::sqrt(x);

  double S_n = 1.0;
  if (spin >= 0) {
    if (f1.spin >= 0 && f2.spin >= 0) { S_n = (f1.spin + 1) * (f2.spin + 1); }
  }
  const double G_n = (i1 == i2) ? 0.5 : 1.0;
  return A * mass_factor * S_n * G_n * std::sqrt(ekin);
}

/// G4FermiFragmentsPoolVI::IsPhysical - is there a pool fragment with this (Z, A) at all.
__host__ __device__ inline bool fermi_is_physical(const FermiPool& pool, int Z, int A) {
  if (A < 0 || A >= kFermiMaxA) { return false; }
  for (int k = 0; k < pool.by_a_count[A]; ++k) {
    if (pool.frag[pool.by_a[pool.by_a_offset[A] + k]].z == Z) { return true; }
  }
  return false;
}

/// G4FermiFragmentsPoolVI::HasChannels. The `exc >` is strict: a fragment sitting exactly on a
/// tabulated level cannot break up from it.
__host__ __device__ inline bool fermi_has_channels(const FermiPool& pool, int Z, int A,
                                                    double exc) {
  if (A < 0 || A >= kFermiMaxA) { return false; }
  for (int j = 0; j < pool.by_a_count[A]; ++j) {
    const int k = pool.by_a_offset[A] + j;
    const FermiFragment& f = pool.frag[pool.by_a[k]];
    if (f.z == Z && exc > f.excitation && pool.ch_count[k] > 0) { return true; }
  }
  return false;
}

/// G4FermiBreakUpVI::IsApplicable. This is the gate the handler and G4Evaporation both consult
/// before doing anything else, so it is the single most load-bearing predicate in the module:
/// a fragment it accepts never reaches the evaporation channels.
__host__ __device__ inline bool fermi_is_applicable(const FermiPool& pool, int Z, int A,
                                                     double eexc) {
  return Z < kFermiMaxZ && A < kFermiMaxA && A > 0 &&
         eexc <= deex_params().fbu_energy_limit && fermi_has_channels(pool, Z, A, eexc);
}

/// G4FermiFragmentsPoolVI::ClosestChannels - the channel-set index, or -1.
///
/// Two ways to win, and the second is not "nearest": an excitation within the 10 eV tolerance
/// of a pool fragment's takes that fragment immediately, and otherwise the fragment with the
/// SMALLEST NON-NEGATIVE `e - E_frag + tolerance` is taken - so only fragments at or below the
/// given mass compete, and the closest one from below wins. A mass below every pool fragment's
/// gives -1.
__host__ __device__ inline int fermi_closest_channels(const FermiPool& pool, int Z, int A,
                                                       double e) {
  if (A < 0 || A >= kFermiMaxA) { return -1; }
  int res = -1;
  double demax = 1.e+9;
  const double tolerance = deex_params().min_excitation;
  for (int j = 0; j < pool.by_a_count[A]; ++j) {
    const int k = pool.by_a_offset[A] + j;
    const FermiFragment& f = pool.frag[pool.by_a[k]];
    if (f.z != Z) { continue; }
    double de = e - f.total_energy();
    if (std::fabs(de) <= tolerance) { return k; }
    de += tolerance;
    if (de >= 0.0 && de <= demax) {
      res = k;
      demax = de;
    }
  }
  return res;
}

/// G4FermiChannels::SamplePair - the first channel whose cumulative probability is at or above
/// `rand`. Returns -1 when none is, which Geant4 signals with a null pointer and treats as
/// "no decay"; the pool's normalisation makes the last entry exactly 1.0, so it cannot happen.
__host__ __device__ inline int fermi_sample_pair(const FermiPool& pool, int k, double rand) {
  for (int i = 0; i < pool.ch_count[k]; ++i) {
    if (rand <= pool.cum_prob[pool.ch_offset[k] + i]) {
      return pool.pair_of_channel[pool.ch_offset[k] + i];
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------------------------
// G4FermiBreakUpVI
// ---------------------------------------------------------------------------------------------

/// The work list BreakFragment grows. Kept on the caller's stack rather than in a vector,
/// because a kernel cannot allocate and because Geant4's own is capped at 100 iterations
/// anyway.
struct FermiWork {
  int frag_index[kFermiMaxWork];
  LorentzVector lvect[kFermiMaxWork];
  int n = 0;
};

/// One state of G4FermiBreakUpVI::SampleDecay. The class keeps these as members and
/// BreakFragment overwrites them per work item; they are grouped so the port cannot read one
/// item's Z with another item's mass.
struct FermiDecayState {
  int z = 0, a = 0, spin = 0;
  double mass = 0.0;
  double excitation = 0.0;   ///< the PRIMARY's, never updated - see the file header
  LorentzVector lv0;
};

/// G4FermiBreakUpVI::SampleDecay. Returns false when the fragment is final; otherwise appends
/// the two products to `w` and returns true.
template <typename Rng>
__host__ __device__ inline bool fermi_sample_decay(const FermiPool& pool, FermiDecayState& s,
                                                    FermiWork& w, Rng& rng) {
  const int k = fermi_closest_channels(pool, s.z, s.a, s.mass);
  if (k < 0) { return false; }
  const int nn = pool.ch_count[k];
  if (nn == 0) { return false; }
  if (w.n + 2 > kFermiMaxWork) { return false; }

  int pair = -1;
  if (nn == 1) {
    pair = pool.pair_of_channel[pool.ch_offset[k]];
  } else if (std::fabs(s.excitation - pool.frag[pool.by_a[k]].excitation) <
             fermi_breakup_tolerance()) {
    // The pool's static probabilities, normalised at initialisation.
    pair = fermi_sample_pair(pool, k, rng.uniform());
  } else {
    // Recomputed, with spin = -1 so the spin factor is dropped, and NOT normalised - the total
    // is multiplied into the random number instead. The last channel is taken unconditionally
    // when the scan falls off the end (`i+1 == nn`), which is what keeps a zero total from
    // returning nothing.
    double ptot = 0.0;
    for (int i = 0; i < nn; ++i) {
      const FermiPair& p = pool.pairs[pool.pair_of_channel[pool.ch_offset[k] + i]];
      ptot += fermi_decay_probability(pool, s.a, -1, s.mass, p.f1, p.f2);
    }
    ptot *= rng.uniform();
    double run = 0.0;
    for (int i = 0; i < nn; ++i) {
      const FermiPair& p = pool.pairs[pool.pair_of_channel[pool.ch_offset[k] + i]];
      run += fermi_decay_probability(pool, s.a, -1, s.mass, p.f1, p.f2);
      if (ptot <= run || i + 1 == nn) {
        pair = pool.pair_of_channel[pool.ch_offset[k] + i];
        break;
      }
    }
  }
  if (pair < 0) { return false; }

  const FermiPair& p = pool.pairs[pair];
  const double mass1 = pool.frag[p.f1].total_energy();
  const double mass2 = pool.frag[p.f2].total_energy();

  double e1 = 0.5 * (s.mass * s.mass - mass2 * mass2 + mass1 * mass1) / s.mass;
  double p1 = 0.0;
  if (e1 > mass1) {
    p1 = std::sqrt((e1 - mass1) * (e1 + mass1));
  } else {
    e1 = mass1;
  }
  const Vec3d v = random_direction(rng);
  LorentzVector lv1(v * p1, e1);

  const Vec3d boost = s.lv0.boost_vector();
  lv1.boost(boost);
  LorentzVector rest = s.lv0;
  rest -= lv1;
  // The second fragment's energy is whatever is left, floored at its mass - and the floor
  // discards the momentum with it, which is Geant4's own `lv0.set(0,0,0,mass2)`.
  if (rest.e < mass2) { rest = LorentzVector(0.0, 0.0, 0.0, mass2); }

  w.frag_index[w.n] = p.f1;
  w.lvect[w.n] = lv1;
  ++w.n;
  w.frag_index[w.n] = p.f2;
  w.lvect[w.n] = rest;
  ++w.n;
  return true;
}

/// What one BreakFragment produced.
struct FermiResult {
  bool consumed_input = false;   ///< whether the input fragment decayed (Geant4 deletes it)
  int n = 0;
  Fragment out[kFermiMaxWork];
  bool refused_work_overflow = false;
};

/// G4FermiBreakUpVI::BreakFragment.
///
/// The 100-iteration cap is Geant4's and it is a `break` AFTER the item has been handled, so
/// the work list can hold items that are never visited; they are silently dropped, and with
/// them their energy. The port keeps the cap and reports the overflow of its own fixed work
/// array separately, because that one is the port's limit and not Geant4's.
template <typename Rng>
__host__ __device__ inline FermiResult fermi_break_fragment(const FermiPool& pool,
                                                             const Fragment& nucleus, Rng& rng) {
  FermiResult res;
  FermiWork w;
  FermiDecayState s;
  s.z = nucleus.z;
  s.a = nucleus.a;
  s.excitation = nucleus.excitation;
  s.mass = nucleus.ground_state_mass + s.excitation;
  s.spin = -1;
  s.lv0 = nucleus.momentum;

  if (!fermi_sample_decay(pool, s, w, rng)) { return res; }
  res.consumed_input = true;

  const int imax = 100;
  for (int i = 0; i < w.n; ++i) {
    const FermiFragment& f = pool.frag[w.frag_index[i]];
    s.z = f.z;
    s.a = f.a;
    s.spin = f.spin;
    s.mass = f.total_energy();
    s.lv0 = w.lvect[i];
    const int before = w.n;
    if (!fermi_sample_decay(pool, s, w, rng)) {
      if (before == w.n && w.n + 2 > kFermiMaxWork) { res.refused_work_overflow = true; }
      if (res.n < kFermiMaxWork) {
        res.out[res.n] = make_fragment(f.a, f.z, w.lvect[i]);
        ++res.n;
      }
    }
    if (i == imax) { break; }
  }
  return res;
}

// ---------------------------------------------------------------------------------------------
// Host-side pool construction - G4FermiFragmentsPoolVI::Initialise
// ---------------------------------------------------------------------------------------------

/// Owns the vectors FermiPool points into.
struct FermiPoolStorage {
  std::vector<FermiFragment> frag;
  std::vector<int> by_a, by_a_offset, by_a_count;
  std::vector<int> ch_offset, ch_count;
  std::vector<FermiPair> pairs;
  std::vector<int> pair_of_channel;
  std::vector<double> cum_prob;

  FermiPool view() const {
    FermiPool p;
    p.frag = frag.data();
    p.n_frag = static_cast<int>(frag.size());
    p.by_a = by_a.data();
    p.by_a_offset = by_a_offset.data();
    p.by_a_count = by_a_count.data();
    p.ch_offset = ch_offset.data();
    p.ch_count = ch_count.data();
    p.pairs = pairs.data();
    p.pair_of_channel = pair_of_channel.data();
    p.cum_prob = cum_prob.data();
    p.n_channels = static_cast<int>(pair_of_channel.size());
    p.n_pairs = static_cast<int>(pairs.size());
    return p;
  }
};

/// G4FermiFragmentsPoolVI::Initialise.
///
/// The order of `frag` is fragment_pool's order, and it matters twice over: list_f[A] is built
/// by scanning it, so the channel sets are indexed by it, and the pair loop's
/// `A2 < A1 || (A2 == A1 && Z2 < Z1)` skip is an ordering on the pool and not on the nuclides.
inline void build_fermi_pool(FermiPoolStorage& st, const data::LevelTable& lt) {
  st = FermiPoolStorage();

  auto push = [&st](int A, int Z, int spin, double exc) {
    FermiFragment f;
    f.a = A;
    f.z = Z;
    f.spin = spin;
    f.excitation = exc;
    f.fragment_mass = deex::nuclear_mass(A, Z);
    st.frag.push_back(f);
  };

  // The eight hard-coded entries, in Geant4's order. He5 (5, 2) and Li5 (5, 3) are unbound and
  // are in the pool anyway: the model needs them as intermediate states.
  push(1, 0, 1, 0.0);
  push(1, 1, 1, 0.0);
  push(2, 1, 2, 0.0);
  push(3, 1, 1, 0.0);
  push(3, 2, 1, 0.0);
  push(4, 2, 0, 0.0);
  push(5, 2, 3, 0.0);
  push(5, 3, 3, 0.0);

  const double tolerance = deex_params().min_excitation;
  const double elim = deex_params().fbu_energy_limit;
  const float elimf = static_cast<float>(elim);

  for (int Z = 1; Z < kFermiMaxZ; ++Z) {
    const int amin = data::level_min_a(Z);
    int amax = data::level_max_a(Z) + 1;
    if (amax > kFermiMaxA) { amax = kFermiMaxA; }
    for (int A = amin; A < amax; ++A) {
      const int m = data::find_manager(lt, Z, A);
      if (m < 0) { continue; }
      // The COMPILED (PhotonEvaporation 5.2) maximum level energy, as a float, and level 0's
      // lifetime as a float. Both exactly zero means "very unstable state" and the whole
      // nuclide is left out of the pool. This is the second place the 5.2/5.7 disagreement of
      // the previous commit is load-bearing: 166 nuclides have a compiled maximum of zero
      // where the installed data has levels, and for those this test turns on the lifetime
      // alone.
      const float compiled_max = data::compiled_max_level_energy_f(Z, A);
      const float t0 = static_cast<float>(data::level_lifetime(lt, m, 0));
      if (compiled_max == 0.0f && t0 == 0.0f) { continue; }

      const int nlev = lt.managers[m].n_levels;
      for (int i = 0; i < nlev; ++i) {
        const float exc = static_cast<float>(data::level_energy(lt, m, i));
        if (exc >= elimf) { continue; }
        const double excd = static_cast<double>(exc);
        bool dup = false;
        for (const FermiFragment& f : st.frag) {
          if (f.z == Z && f.a == A && std::fabs(excd - f.excitation) < tolerance) {
            dup = true;
            break;
          }
        }
        if (dup) { continue; }
        push(A, Z, data::level_spin_two(lt, m, i), excd);
      }
    }
  }

  // list_f[A] / list_c[A], in fragment_pool order.
  const int nfrag = static_cast<int>(st.frag.size());
  st.by_a_offset.assign(kFermiMaxA, 0);
  st.by_a_count.assign(kFermiMaxA, 0);
  for (int A = 0; A < kFermiMaxA; ++A) {
    st.by_a_offset[A] = static_cast<int>(st.by_a.size());
    for (int i = 0; i < nfrag; ++i) {
      if (st.frag[i].a == A) { st.by_a.push_back(i); }
    }
    st.by_a_count[A] = static_cast<int>(st.by_a.size()) - st.by_a_offset[A];
  }
  const int nsets = static_cast<int>(st.by_a.size());

  // Channel sets, built as vectors first because the pair loop appends to them out of order.
  std::vector<std::vector<int>> sets(static_cast<std::size_t>(nsets));
  // list_p[A], and the pair index for each (f1, f2) already added under that A.
  std::vector<std::vector<int>> pairs_by_a(kFermiMaxA);

  for (int i = 0; i < nfrag; ++i) {
    const FermiFragment& f1 = st.frag[i];
    for (int j = 0; j < nfrag; ++j) {
      const FermiFragment& f2 = st.frag[j];
      if (f2.a < f1.a || (f2.a == f1.a && f2.z < f1.z)) { continue; }
      const int Z = f1.z + f2.z;
      const int A = f1.a + f2.a;
      if (Z >= kFermiMaxZ || A >= kFermiMaxA) { continue; }
      // IsInPhysPairs, by index rather than by pointer.
      bool seen = false;
      for (int pi : pairs_by_a[A]) {
        if (st.pairs[pi].f1 == i && st.pairs[pi].f2 == j) {
          seen = true;
          break;
        }
      }
      if (seen) { continue; }

      double minE = f1.total_energy() + f2.total_energy();
      double exc = 0.0;
      if (fermi_is_physical(st.view(), Z, A)) {
        minE += deex::coulomb_barrier(f1.a, f1.z, f2.a, f2.z, 0.0);
        exc = minE - deex::nuclear_mass(A, Z);
      }
      if (exc >= elim) { continue; }

      int pair_index = -1;
      for (int k = 0; k < st.by_a_count[A]; ++k) {
        const int slot = st.by_a_offset[A] + k;
        const FermiFragment& f3 = st.frag[st.by_a[slot]];
        if (Z == f3.z && f3.total_energy() - minE + tolerance >= 0.0) {
          if (pair_index < 0) {
            pair_index = static_cast<int>(st.pairs.size());
            st.pairs.push_back(FermiPair{i, j});
            pairs_by_a[A].push_back(pair_index);
          }
          sets[static_cast<std::size_t>(slot)].push_back(pair_index);
        }
      }
    }
  }

  // Flatten, then compute the static probabilities exactly as the pool does - including that a
  // single-channel set keeps the 1.0 AddChannel pushed, and that a set whose total probability
  // is zero gets prob[0] = 1.0 and leaves the rest at 1.0 too.
  st.ch_offset.assign(static_cast<std::size_t>(nsets), 0);
  st.ch_count.assign(static_cast<std::size_t>(nsets), 0);
  for (int k = 0; k < nsets; ++k) {
    st.ch_offset[k] = static_cast<int>(st.pair_of_channel.size());
    for (int p : sets[static_cast<std::size_t>(k)]) {
      st.pair_of_channel.push_back(p);
      st.cum_prob.push_back(1.0);
    }
    st.ch_count[k] = static_cast<int>(st.pair_of_channel.size()) - st.ch_offset[k];
  }

  const FermiPool pool = st.view();
  for (int k = 0; k < nsets; ++k) {
    const int nch = st.ch_count[k];
    if (nch <= 1) { continue; }
    const FermiFragment& f = st.frag[st.by_a[k]];
    double ptot = 0.0;
    for (int i = 0; i < nch; ++i) {
      const FermiPair& p = st.pairs[st.pair_of_channel[st.ch_offset[k] + i]];
      ptot += fermi_decay_probability(pool, f.a, f.spin, f.total_energy(), p.f1, p.f2);
      st.cum_prob[st.ch_offset[k] + i] = ptot;
    }
    if (ptot == 0.0) {
      st.cum_prob[st.ch_offset[k]] = 1.0;
    } else {
      const double inv = 1.0 / ptot;
      for (int i = 0; i < nch - 1; ++i) { st.cum_prob[st.ch_offset[k] + i] *= inv; }
      st.cum_prob[st.ch_offset[k] + nch - 1] = 1.0;
    }
  }
}

}  // namespace g4gpu::deex

#endif
