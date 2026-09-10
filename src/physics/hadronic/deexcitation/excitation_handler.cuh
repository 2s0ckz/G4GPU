// G4ExcitationHandler: the entry point of the whole module, and the device-callable form of it.
//
// Transcribed from G4ExcitationHandler, G4Evaporation and G4UnstableFragmentBreakUp (11.1.1).
//
// One excited fragment in, a list of fragments and gammas out. The dispatch, with 11.1.1's
// defaults, is three nested decisions and every one of them is a threshold rather than a model
// choice:
//
//   BreakItUp        A <= 1, or a cold natural isotope       -> released as it is
//                    A < 17 and Z < 9, or E*/A <= 200 GeV    -> the evaporation list
//                    otherwise                               -> G4StatMF, REFUSED by name
//   the loop         Fermi break-up applicable                -> two-body cascade, and if it
//                                                               made more than one fragment
//                                                               that is the whole answer
//                    otherwise                                -> G4Evaporation::BreakFragment
//   BreakFragment    photon evaporation holds ALL the         -> BreakUpChain, walk the gamma
//                    probability                                cascade to the ground state
//                    no channel is open and A < 30            -> G4UnstableFragmentBreakUp
//                    otherwise                                -> sample one of 68 channels
//
// The multifragmentation branch is unreachable: `fMinExPerNucleounForMF` is 200 GeV per
// nucleon (see the first commit of this package), so G4StatMF is not ported. The condition is
// reproduced anyway and reports a refusal by name, so that a caller who ever raises E*/A past
// the limit is told rather than quietly evaporated.
//
// **Where this stops, and why.** The output is a list of (Z, A, E*, floating level,
// four-momentum) and a PDG code for the things that have an unambiguous one - a gamma, a
// conversion electron, and the six light nuclei G4ExcitationHandler maps to fixed particle
// definitions. It does NOT produce the PDG code of an excited heavy ion, and that is a refusal
// rather than an omission: G4ExcitationHandler's last loop calls
// `G4IonTable::GetIon(Z, A, E*, FloatLevelBase(idx))`, which snaps E* to the G4ENSDFSTATE
// isomer table, takes the isomer level from there, and puts that level into the last digit of
// the encoding - and which level index an ion gets depends on the order ions were created in
// during the run. So the code is a property of the run's ion table, not of the fragment. The
// mapping belongs to whoever owns core/particle.cuh; this module hands over the physics.
// 2.3% of the products in ref/oracle/deex_breakup.csv carry a non-zero isomer digit, and their
// masses - and therefore their kinetic energies - are the ones this boundary affects.
//
// Buffers are the caller's. Three of them: the evaporation work list, the finished-fragment
// list, and the products. G4ExcitationHandler reserves 30, 60 and 30 and grows them; a kernel
// cannot, so the capacities are passed in and an overflow is reported by name instead of
// being wrapped around. The evaporation list is the one that grows: Geant4's own guard is a
// fatal exception at 1000 iterations over it.
#ifndef G4GPU_DEEX_EXCITATION_HANDLER_CUH
#define G4GPU_DEEX_EXCITATION_HANDLER_CUH

#include <cmath>

#include "core/units.cuh"
#include "data/level_data.cuh"
#include "data/natural_isotopes.hh"
#include "physics/hadronic/deexcitation/corrections.cuh"
#include "physics/hadronic/deexcitation/deex_params.cuh"
#include "physics/hadronic/deexcitation/evaporation.cuh"
#include "physics/hadronic/deexcitation/fermi_breakup.cuh"
#include "physics/hadronic/deexcitation/fission.cuh"
#include "physics/hadronic/deexcitation/fragment.cuh"
#include "physics/hadronic/deexcitation/gem.cuh"
#include "physics/hadronic/deexcitation/nuclear_masses.cuh"
#include "physics/hadronic/deexcitation/photon_evaporation.cuh"

namespace g4gpu::deex {

namespace u = g4gpu::units;

/// The 68 channels of G4EvaporationDefaultGEMFactory, and where each index lives. The layout
/// is observable: G4Evaporation::BreakFragment treats index 0 specially (it is the one whose
/// probability being the whole total starts a gamma cascade) and only allows its early exit
/// from index 8, which is the first GEM channel.
constexpr int kNumDeexChannels = 68;
constexpr int kChannelPhoton = 0;
constexpr int kChannelFission = 1;
constexpr int kChannelFirstEvaporation = 2;   ///< n, p, d, t, He3, alpha at 2..7
constexpr int kChannelFirstGem = 8;           ///< the sixty GEM nuclei at 8..67

/// One product. `pdg` is filled where G4ExcitationHandler uses a fixed particle definition and
/// left 0 where it would have asked G4IonTable for an ion - see the file header.
struct DeexProduct {
  LorentzVector momentum;
  int z = 0;
  int a = 0;
  int pdg = 0;
  double excitation = 0.0;   ///< already zeroed below fMinExcitation, as the handler does
  int floating_level = 0;
  bool long_lived = false;
};

/// PDG codes for the eight species G4ExcitationHandler resolves without the ion table.
__host__ __device__ inline int deex_fixed_pdg(int Z, int A, int pdg_if_not_nucleus) {
  if (A == 0) { return pdg_if_not_nucleus; }             // gamma or conversion electron
  if (A == 1 && Z == 0) { return 2112; }                 // neutron
  if (A == 1 && Z == 1) { return 2212; }                 // proton
  if (A == 2 && Z == 1) { return 1000010020; }           // deuteron
  if (A == 3 && Z == 1) { return 1000010030; }           // triton
  if (A == 3 && Z == 2) { return 1000020030; }           // He3
  if (A == 4 && Z == 2) { return 1000020040; }           // alpha
  return 0;                                              // an ion: G4IonTable's job
}

/// Caller-owned scratch. Sizes are the caller's contract; `evap_capacity` is the one that has
/// to be generous, because SortSecondaryFragment pushes every hot or unstable product back
/// onto it and Geant4's own limit on the loop over it is 1000.
///
/// `step` holds what ONE call of G4Evaporation::BreakFragment emitted. Its loop runs at most A
/// times and pushes at most one fragment per pass, plus the gammas of a BreakUpChain, so it is
/// bounded by A plus the length of the longest gamma cascade. It is a caller buffer rather than
/// a local array because a device function's stack is not the place for it.
struct DeexWorkspace {
  Fragment* evap_list = nullptr;
  int evap_capacity = 0;
  Fragment* results = nullptr;
  int results_capacity = 0;
  Fragment* step = nullptr;
  int step_capacity = 0;
  DeexProduct* products = nullptr;
  int products_capacity = 0;
};

/// CLHEP's `Hep3Vector::unit()`, which returns the ZERO VECTOR for a zero vector - not a
/// default axis. src/core/vec3.cuh's `normalize` returns (0, 0, 1) instead, which is the safer
/// choice everywhere else in this port and the wrong one here: G4UnstableFragmentBreakUp takes
/// the unit vector of a fragment's momentum, and a fragment at rest - which is most of the
/// validated grid - must stay at rest rather than acquire momentum along z.
__host__ __device__ inline Vec3d clhep_unit(const Vec3d& v) {
  const double tot = g4gpu::mag2(v);
  if (tot <= 0.0) { return v; }
  const double inv = 1.0 / std::sqrt(tot);
  return Vec3d{v.x * inv, v.y * inv, v.z * inv};
}

/// What the call did, and every refusal it hit, by name.
struct DeexStatus {
  int n_products = 0;
  int n_evaporation_steps = 0;

  /// E*/A above fMinExPerNucleounForMF: G4StatMF would run. Not ported.
  bool refused_multifragmentation = false;
  /// A fragment with lambdas != 0. G4ExcitationHandler's hyper-nucleus path is not ported.
  bool refused_hyper_fragment = false;
  /// G4UnstableFragmentBreakUp found no decay channel at all - its `idx` stayed -1 and Geant4
  /// would index its tables with it.
  bool refused_unstable_no_channel = false;
  /// G4CompetitiveFission's 100 failed splits, where Geant4 throws.
  bool refused_fission_split = false;
  /// One of the three buffers was too small.
  bool refused_capacity = false;
  /// The handler's own 1000-iteration guard on the evaporation list, where Geant4 aborts.
  bool refused_loop_limit = false;
  /// A fragment whose 4-momentum came out spacelike enough to give a negative excitation.
  bool negative_excitation = false;

  int refused_z = 0, refused_a = 0;   ///< the fragment a refusal names

  __host__ __device__ bool any_refusal() const {
    return refused_multifragmentation || refused_hyper_fragment ||
           refused_unstable_no_channel || refused_fission_split || refused_capacity ||
           refused_loop_limit;
  }
};

// ---------------------------------------------------------------------------------------------
// G4UnstableFragmentBreakUp
// ---------------------------------------------------------------------------------------------

/// The six decay products G4UnstableFragmentBreakUp can emit: n, p, d, t, He3, alpha. Same six
/// as the evaporation channels, in the same order, but this is a different table in a different
/// class and it is written out rather than shared - G4UnstableFragmentBreakUp::Zfr/Afr.
__host__ __device__ inline const int* unstable_zfr() {
  static const int z[6] = {0, 1, 1, 1, 2, 2};
  return z;
}
__host__ __device__ inline const int* unstable_afr() {
  static const int a[6] = {1, 1, 2, 3, 3, 4};
  return a;
}

/// G4UnstableFragmentBreakUp::BreakUpChain - despite the name it emits ONE fragment and
/// returns; G4Evaporation::BreakFragment calls it again on the next pass.
///
/// The channel search is a maximisation, not a scan for the first that works: it walks the six
/// ejectiles, and for each computes the mass left over, keeping whichever leaves the MOST -
/// and breaks out as soon as one leaves a positive amount. So a fragment for which nothing is
/// energetically allowed still comes out with `idx` naming the least-bad channel, and the code
/// below then forces the decay at threshold if it is within 5 keV.
///
/// Two sub-cases inside the loop. When the residual has four nucleons or fewer, it must ITSELF
/// be one of the six, so the inner loop over `j` looks it up and both masses are table
/// entries; otherwise the residual's mass comes from G4NucleiProperties and, when the residual
/// is heavier than an alpha and there is energy to spare, it is given a UNIFORMLY RANDOM
/// excitation between zero and that surplus. That random draw is the only one in the function
/// before the direction.
///
/// **REFUSED:** when no ejectile satisfies the (Zres, Ares) conditions at all - which needs
/// Z > A, i.e. an unphysical fragment - Geant4 leaves `idx` at -1 and then reads `Zfr[-1]`.
/// The port reports the refusal instead.
template <typename Rng>
__host__ __device__ inline bool unstable_break_up(Fragment& nucleus, Fragment& emitted,
                                                   DeexStatus& st, Rng& rng) {
  int Z = nucleus.z;
  int A = nucleus.a;
  LorentzVector lv = nucleus.momentum;

  const double tolerance = 10.0 * u::eV<double>();
  const double dmlimit = 0.005 * u::MeV<double>();
  double mass = lv.mag();
  double exca = -1000.0;
  bool is_channel = false;
  int idx = -1;
  double mass1 = 0.0, mass2 = 0.0;

  for (int i = 0; i < 6; ++i) {
    const int zi = unstable_zfr()[i];
    const int ai = unstable_afr()[i];
    const int zres = Z - zi;
    const int ares = A - ai;
    if (!(zres >= 0 && ares >= zres && ares >= ai)) { continue; }

    if (ares <= 4) {
      for (int j = 0; j < 6; ++j) {
        if (zres == unstable_zfr()[j] && ares == unstable_afr()[j]) {
          const double delm = mass - deex::nuclear_mass(ai, zi) -
                              deex::nuclear_mass(unstable_afr()[j], unstable_zfr()[j]);
          if (delm > exca) {
            mass2 = deex::nuclear_mass(ai, zi);                                    // emitted
            mass1 = deex::nuclear_mass(unstable_afr()[j], unstable_zfr()[j]);       // recoil
            exca = delm;
            idx = i;
            if (delm > 0.0) {
              is_channel = true;
              break;
            }
          }
        }
      }
    }
    if (is_channel) { break; }
    const double mres = deex::nuclear_mass(ares, zres);
    const double e = mass - mres - deex::nuclear_mass(ai, zi);
    if (e >= exca) {
      mass2 = deex::nuclear_mass(ai, zi);
      mass1 = (ares > 4 && e > 0.0) ? mres + e * rng.uniform() : mres;
      exca = e;
      idx = i;
      if (e > 0.0) {
        is_channel = true;
        break;
      }
    }
  }

  if (idx < 0) {
    st.refused_unstable_no_channel = true;
    st.refused_z = Z;
    st.refused_a = A;
    return false;
  }

  const double massmin = mass1 + mass2;
  if (!is_channel || mass < massmin) {
    if (mass + dmlimit < massmin) { return false; }
    // Forced at threshold: the invariant mass is raised to massmin and the momentum rebuilt
    // from the ORIGINAL total energy, so the decay conserves energy and breaks momentum
    // conservation by up to 5 keV. That is Geant4's trade and the 5 keV is `dmlimit`.
    mass = massmin;
    double e = lv.e;
    if (e < mass + tolerance) { e = mass + tolerance; }
    const double mom = std::sqrt((e - mass) * (e + mass));
    const Vec3d dir = clhep_unit(lv.v);
    lv = LorentzVector(dir * mom, e);
  }

  double e2 = 0.5 * ((mass - mass1) * (mass + mass1) + mass2 * mass2) / mass;
  if (e2 < mass2) { e2 = mass2; }
  const double mom = std::sqrt((e2 - mass2) * (e2 + mass2));

  const Vec3d bst = lv.boost_vector();
  const Vec3d v = random_direction(rng);
  LorentzVector mom2(v * mom, e2);
  mom2.boost(bst);
  emitted = make_fragment(unstable_afr()[idx], unstable_zfr()[idx], mom2);

  lv -= mom2;
  Z -= unstable_zfr()[idx];
  A -= unstable_afr()[idx];
  nucleus.set_za_and_momentum(lv, Z, A);
  return true;
}

// ---------------------------------------------------------------------------------------------
// The 68 channels, behind one probability call and one emission call
// ---------------------------------------------------------------------------------------------

/// Everything one channel's GetEmissionProbability leaves behind. Geant4 keeps 68 of these
/// alive - one per channel object - and EmittedFragment reads whichever the sampled channel
/// left. The port keeps ONE and recomputes it for the sampled channel, which is exactly
/// equivalent and not a shortcut: every one of the three state functions writes every field it
/// later reads, and none of them draws a random number. The one exception is photon
/// evaporation, whose `level_index` is a search hint that must survive from one gamma to the
/// next; that state is therefore held for the whole BreakItUp call, as G4ExcitationHandler
/// holds one G4PhotonEvaporation.
struct ChannelState {
  EvaporationState evap;
  GemState gem;
  FissionState fission;
};

/// GetEmissionProbability for channel `i`. Photon evaporation is index 0 and takes the
/// persistent state; everything else takes the scratch one.
__host__ __device__ inline double deex_channel_probability(int i, const Fragment& frag,
                                                            const data::LevelTable& lt,
                                                            PhotonEvaporationState& ps,
                                                            ChannelState& cs) {
  if (i == kChannelPhoton) { return photon_emission_probability(ps, frag, lt); }
  if (i == kChannelFission) { return fission_emission_probability(cs.fission, frag, lt); }
  if (i < kChannelFirstGem) {
    cs.evap.ej = i - kChannelFirstEvaporation;
    return channel_emission_probability(cs.evap, frag, lt);
  }
  cs.gem.ch = i - kChannelFirstGem;
  return gem_channel_emission_probability(cs.gem, frag, lt);
}

// ---------------------------------------------------------------------------------------------
// G4Evaporation::BreakFragment
// ---------------------------------------------------------------------------------------------

/// A small append-only list over a caller-owned array.
struct FragmentList {
  Fragment* v = nullptr;
  int capacity = 0;
  int n = 0;
  __host__ __device__ bool push(const Fragment& f) {
    if (n >= capacity) { return false; }
    v[n] = f;
    ++n;
    return true;
  }
};

/// G4PhotonEvaporation::BreakUpChain - emit gammas until GenerateGamma returns nothing.
template <typename Rng>
__host__ __device__ inline void photon_break_up_chain(PhotonEvaporationState& ps,
                                                       Fragment& nucleus, FragmentList& out,
                                                       const data::LevelTable& lt,
                                                       DeexStatus& st, Rng& rng) {
  for (;;) {
    const GammaEmission g = generate_gamma(ps, nucleus, lt, rng);
    if (!g.emitted) { return; }
    if (!out.push(g.product)) {
      st.refused_capacity = true;
      st.refused_z = nucleus.z;
      st.refused_a = nucleus.a;
      return;
    }
  }
}

/// G4Evaporation::BreakFragment. `nucleus` is modified into the residual; emitted fragments go
/// to `out`.
///
/// Three details reproduced because each one changes an outcome:
///
///   * `oldprob` is declared OUTSIDE the evaporation loop and set at the end of every channel
///     iteration, so at the start of each step it holds the LAST channel's probability from
///     the previous step. It is half of the early-exit test, and on the first step it is zero
///     by initialisation rather than by meaning.
///   * the early exit is allowed only from channel index 8 - the first GEM channel - and only
///     when the current probability is positive. So no exit can happen inside photon
///     evaporation, fission or the six light ejectiles however small their probabilities are.
///   * `probabilities[0] == totprob` is an exact double comparison, and it is how "photon
///     evaporation is the only open channel" is detected. Adding a tolerance would start
///     gamma cascades for fragments Geant4 evaporates a nucleon from.
template <typename Rng>
__host__ __device__ inline void evaporation_break_fragment(Fragment& nucleus, FragmentList& out,
                                                            const data::LevelTable& lt,
                                                            const FermiPool& pool,
                                                            PhotonEvaporationState& ps,
                                                            DeexStatus& st, Rng& rng) {
  const int amax = nucleus.a;
  const double min_excitation = evaporation_min_excitation();
  double probabilities[kNumDeexChannels];
  double oldprob = 0.0;

  for (int ia = 0; ia < amax; ++ia) {
    const int Z = nucleus.z;
    const int A = nucleus.a;
    if (A <= 1) { return; }
    double eex = nucleus.excitation;
    if (fermi_is_applicable(pool, Z, A, eex)) { return; }

    const bool abun = data::is_natural_isotope(Z, A);
    if (eex <= min_excitation && (abun || (A == 3 && (Z == 1 || Z == 2)))) { return; }

    ++st.n_evaporation_steps;
    double totprob = 0.0;
    int maxchannel = kNumDeexChannels;
    ChannelState cs;
    for (int i = 0; i < kNumDeexChannels; ++i) {
      const double prob = deex_channel_probability(i, nucleus, lt, ps, cs);
      totprob += prob;
      probabilities[i] = totprob;
      if (i >= 8 && prob > 0.0) {
        if (prob <= totprob * 1.e-8 && oldprob <= totprob * 1.e-8) {
          maxchannel = i + 1;
          break;
        }
      }
      oldprob = prob;
    }

    if (0.0 < totprob && probabilities[0] == totprob) {
      photon_break_up_chain(ps, nucleus, out, lt, st, rng);
      if (st.refused_capacity) { return; }
      if (abun) {
        nucleus.long_lived = true;
        return;
      }
      eex = nucleus.excitation;
      if (fermi_is_applicable(pool, Z, A, eex)) { return; }
      if (nucleus.long_lived) { return; }
      totprob = 0.0;
    }

    if (0.0 == totprob && A < 30) {
      Fragment emitted;
      if (unstable_break_up(nucleus, emitted, st, rng)) {
        if (!out.push(emitted)) {
          st.refused_capacity = true;
          return;
        }
        continue;
      }
      return;
    }

    // Geant4's own scan, including that it can fall off the end: `for(i=0;i<maxchannel;++i)
    // { if(probabilities[i] >= totprob) break; }` leaves i == maxchannel when nothing matched
    // and then indexes theChannels with it. It cannot happen - probabilities[maxchannel-1] is
    // the unscaled total and rng.uniform() is below 1 - but the port refuses rather than
    // reading past the end, because "cannot happen" is a claim about this arithmetic.
    totprob *= rng.uniform();
    int chosen = 0;
    for (chosen = 0; chosen < maxchannel; ++chosen) {
      if (probabilities[chosen] >= totprob) { break; }
    }
    if (chosen >= maxchannel) {
      st.refused_capacity = true;
      st.refused_z = Z;
      st.refused_a = A;
      return;
    }

    // Recompute the chosen channel's state, then emit. See ChannelState's comment for why
    // this is the same thing as keeping all 68.
    Fragment emitted;
    bool have = false;
    if (chosen == kChannelPhoton) {
      const GammaEmission g = generate_gamma(ps, nucleus, lt, rng);
      have = g.emitted;
      emitted = g.product;
    } else if (chosen == kChannelFission) {
      ChannelState fs;
      fission_emission_probability(fs.fission, nucleus, lt);
      const FissionProducts fp = fission_emitted_fragment(fs.fission, nucleus, rng);
      if (fp.refused_no_valid_split) {
        st.refused_fission_split = true;
        st.refused_z = Z;
        st.refused_a = A;
        return;
      }
      have = fp.fissioned;
      emitted = fp.fragment1;
    } else if (chosen < kChannelFirstGem) {
      ChannelState es;
      es.evap.ej = chosen - kChannelFirstEvaporation;
      channel_emission_probability(es.evap, nucleus, lt);
      emitted = channel_emitted_fragment(es.evap, nucleus, lt, rng);
      have = true;
    } else {
      ChannelState gs;
      gs.gem.ch = chosen - kChannelFirstGem;
      gem_channel_emission_probability(gs.gem, nucleus, lt);
      emitted = gem_emitted_fragment(gs.gem, nucleus, lt, rng);
      have = true;
    }
    if (!have) { return; }
    if (!out.push(emitted)) {
      st.refused_capacity = true;
      st.refused_z = Z;
      st.refused_a = A;
      return;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// G4ExcitationHandler::BreakItUp
// ---------------------------------------------------------------------------------------------

/// G4ExcitationHandler::SortSecondaryFragment - the one place a fragment is decided to be
/// finished or to go round again.
///
/// `A <= 1 || IsLongLived()` releases it, a COLD fragment is released if it is a natural
/// isotope or one of d/t/He3 (`A == 3 && (Z == 1 || Z == 2)` - which is triton and He3, and
/// note the deuteron is not in this clause because it is a natural isotope anyway), and
/// everything else goes back onto the evaporation list. A hot fragment always goes round again.
__host__ __device__ inline void sort_secondary_fragment(const Fragment& f, FragmentList& results,
                                                         FragmentList& evap_list,
                                                         DeexStatus& st) {
  const int A = f.a;
  bool ok = true;
  if (A <= 1 || f.long_lived) {
    ok = results.push(f);
  } else if (f.excitation < deex_params().min_excitation) {
    const int Z = f.z;
    if (data::is_natural_isotope(Z, A) || (A == 3 && (Z == 1 || Z == 2))) {
      ok = results.push(f);
    } else {
      ok = evap_list.push(f);
    }
  } else {
    ok = evap_list.push(f);
  }
  if (!ok) {
    st.refused_capacity = true;
    st.refused_z = f.z;
    st.refused_a = f.a;
  }
}

/// G4ExcitationHandler::BreakItUp. The device-callable entry point of the package.
///
/// Contract: `initial` is an excited fragment - (Z, A) and a four-momentum whose invariant mass
/// is M(Z, A) + E*, which is the only way G4Fragment accepts an excitation. On return,
/// `ws.products[0 .. status.n_products)` holds the final state and `status` names every
/// refusal. Nothing is allocated and no random numbers are drawn beyond `rng`.
template <typename Rng>
__host__ __device__ inline DeexStatus deexcite(const Fragment& initial,
                                                const data::LevelTable& lt,
                                                const FermiPool& pool,
                                                const DeexWorkspace& ws, Rng& rng) {
  DeexStatus st;
  FragmentList results{ws.results, ws.results_capacity, 0};
  FragmentList evap_list{ws.evap_list, ws.evap_capacity, 0};

  if (initial.lambdas != 0) {
    st.refused_hyper_fragment = true;
    st.refused_z = initial.z;
    st.refused_a = initial.a;
    return st;
  }

  const double ex_energy = initial.excitation;
  const int A = initial.a;
  const int Z = initial.z;
  const double min_excitation = deex_params().min_excitation;

  // One persistent photon-evaporation state for the whole call, matching the lifetime of the
  // single G4PhotonEvaporation the handler owns: its `level_index` is a search hint that must
  // survive from one gamma of a cascade to the next.
  PhotonEvaporationState ps;

  if (A <= 1) {
    if (!results.push(initial)) { st.refused_capacity = true; }
  } else if (ex_energy < min_excitation && data::is_natural_isotope(Z, A)) {
    if (!results.push(initial)) { st.refused_capacity = true; }
  } else if ((A < kMaxAForFermiBreakUp && Z < kMaxZForFermiBreakUp) ||
             ex_energy <= deex_params().min_ex_per_nucleon_for_mf * A) {
    if (!evap_list.push(initial)) { st.refused_capacity = true; }
  } else {
    // G4StatMF. Unreachable with fMinExPerNucleounForMF = 200 GeV per nucleon; refused by
    // name so a caller who raises it is told rather than evaporated.
    st.refused_multifragmentation = true;
    st.refused_z = Z;
    st.refused_a = A;
    return st;
  }

  const int countmax = 1000;
  for (int kk = 0; kk < evap_list.n; ++kk) {
    if (kk >= countmax) {
      st.refused_loop_limit = true;
      st.refused_z = evap_list.v[kk].z;
      st.refused_a = evap_list.v[kk].a;
      break;
    }
    Fragment frag = evap_list.v[kk];
    if (frag.negative_excitation) { st.negative_excitation = true; }

    // Fermi break-up, and it is all-or-nothing: more than one product and that IS the answer
    // for this fragment, one or none and evaporation is applied to it instead.
    if (fermi_is_applicable(pool, frag.z, frag.a, frag.excitation)) {
      const FermiResult fr = fermi_break_fragment(pool, frag, rng);
      if (fr.refused_work_overflow) { st.refused_capacity = true; }
      if (fr.n > 1) {
        for (int i = 0; i < fr.n; ++i) {
          sort_secondary_fragment(fr.out[i], results, evap_list, st);
        }
        if (st.refused_capacity) { break; }
        continue;
      }
      // Geant4 deletes the input fragment when the first decay succeeded, which can only
      // leave fewer than two products if the work list overflowed. `frag` is a copy here, so
      // evaporation is applied to it exactly as Geant4 applies it to the surviving pointer.
    }

    FragmentList step{ws.step, ws.step_capacity, 0};
    evaporation_break_fragment(frag, step, lt, pool, ps, st, rng);
    if (step.n == 0) {
      if (!results.push(frag)) { st.refused_capacity = true; }
    } else {
      sort_secondary_fragment(frag, results, evap_list, st);
    }
    for (int i = 0; i < step.n; ++i) {
      sort_secondary_fragment(step.v[i], results, evap_list, st);
    }
    if (st.refused_capacity) { break; }
  }

  // The final conversion. G4ExcitationHandler resolves eight species to fixed particle
  // definitions and asks G4IonTable for everything else; the ion's excitation is zeroed below
  // fMinExcitation and its floating-level index with it, which is the only part of that last
  // loop that is physics rather than bookkeeping.
  for (int i = 0; i < results.n; ++i) {
    if (st.n_products >= ws.products_capacity) {
      st.refused_capacity = true;
      break;
    }
    const Fragment& f = results.v[i];
    DeexProduct& p = ws.products[st.n_products];
    p.momentum = f.momentum;
    p.z = f.z;
    p.a = f.a;
    p.pdg = deex_fixed_pdg(f.z, f.a, f.pdg_if_not_nucleus);
    p.long_lived = f.long_lived;
    if (p.pdg != 0 || f.a == 0) {
      p.excitation = 0.0;
      p.floating_level = 0;
    } else {
      p.excitation = f.excitation;
      p.floating_level = f.floating_level;
      if (p.excitation < min_excitation) {
        p.excitation = 0.0;
        p.floating_level = 0;
      }
    }
    if (f.negative_excitation) { st.negative_excitation = true; }
    ++st.n_products;
  }
  return st;
}

/// The kinetic energy G4ReactionProduct would report for a product.
///
/// `G4ReactionProduct::SetTotalEnergy` sets `kineticEnergy = totalEnergy - mass` with `mass`
/// taken from the particle definition, so for a nucleus this is the total energy minus the ion
/// mass INCLUDING its excitation. Exact for the 97.7% of products that come out in their
/// ground state; for an isomer, G4IonTable snaps the excitation to the G4ENSDFSTATE table
/// before building the ion, so its mass - and this number - differ by the snapping. See the
/// file header: that snapping is refused by name and belongs to the ion table.
__host__ __device__ inline double deex_kinetic_energy(const DeexProduct& p) {
  double mass;
  if (p.a == 0) {
    mass = (p.pdg == kPdgElectron) ? u::electron_mass_c2<double>() : 0.0;
  } else {
    mass = deex::nuclear_mass(p.a, p.z) + p.excitation;
  }
  const double k = p.momentum.e - mass;
  return (k > 0.0) ? k : 0.0;
}

}  // namespace g4gpu::deex

#endif
