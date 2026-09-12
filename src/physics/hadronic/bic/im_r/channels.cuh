// The 306 concrete NN -> resonance channels, and the two-level buffered cross section that
// decides which of G4CollisionNN's eight components a collision goes to.
//
// Transcribed from G4GeneralNNCollision's four `MakeNNTo*` templates, the six `G4CollisionNNTo*`
// composites and the nine per-multiplet classes under two of them, G4CollisionComposite
// (`BufferCrossSection`, `BufferedCrossSection`, `FinalState`) and G4CrossSectionBuffer
// (im_r_matrix, 11.1.1).
//
// ## The shape of the tree
//
// `G4CollisionNN` has eight components. Two are elastic and answer directly; the other six are
// composites with NO cross-section source of their own, so each answers from a 32-point BUFFER:
//
//     component 2  G4CollisionNNToNDelta          6 concrete   NN -> N Delta(1232)
//     component 3  G4CollisionNNToDeltaDelta      6 concrete   NN -> Delta Delta
//     component 4  G4CollisionNNToNDeltastar      9 children x 6 = 54   NN -> N Delta*
//     component 5  G4CollisionNNToDeltaDeltastar  9 children x 10 = 90  NN -> Delta Delta*
//     component 6  G4CollisionNNToNNstar          15 x 4 = 60           NN -> N N*
//     component 7  G4CollisionNNToDeltaNstar      15 x 6 = 90           NN -> Delta N*
//
// 306 concrete channels. Components 4 and 5 have a MIDDLE layer - nine `G4CollisionNNToNDelta1600`
// -style classes, one per Delta* multiplet, each itself a bufferless composite - so their totals
// are a buffer of a sum of buffers. Components 6 and 7 have no middle layer: their constructors
// call `MakeNNToNNStar`/`MakeNNToDeltaNstar` fifteen times directly into themselves.
//
// The middle layer is not cosmetic. `G4CrossSectionBuffer::CrossSection` ends with
//
//     if(y1<0.01*CLHEP::millibarn) result = 0;
//
// - a floor on the LEFT NODE, not on the result - and that floor is applied once per buffer. A
// Delta* multiplet whose own buffered node is below 0.01 mb contributes exactly zero to its
// parent, where the same channels summed in one buffer would have contributed their sum. So the
// nesting changes the answer, and the port reproduces it.
//
// ## Four things in G4CrossSectionBuffer::CrossSection that are not interpolation
//
//   * **Above the whole grid the cross section is ZERO.** The search loop breaks on the first
//     node past `sqrts`; if there is none it falls out with `x1, y1, x2, y2` still at their
//     declaration values `(1, 0, 2, 0)`, and `result = 0 + (sqrts-1)*(0-0)/(2-1)` is 0. The grid
//     tops out at `theT = 100 GeV`, so nothing in QBBC reaches it - but it is the class's
//     behaviour and not an extrapolation.
//   * **Below the grid it extrapolates BACKWARDS.** The `0 == i` branch uses nodes 0 and 1 and
//     the straight line through them, continued to the left. Every one of these tables is zero at
//     the first few nodes, so the extrapolation runs along zero.
//   * **The 0.01 mb floor is on `y1`.** A point whose left node is below 0.01 mb returns 0 even
//     if the interpolated value would have been larger.
//   * **`theData.size() == 1` returns the single value**, which cannot happen with 32 points.
//
// ## The buffer's grid is built with the kinetic energy on the LIGHTER particle
//
// `BufferCrossSection` puts `theT[i]` on whichever of the two definitions has the smaller PDG
// mass and leaves the other at rest, then stores `(a4Momentum+b4Momentum).mag()` against the sum
// of the components' cross sections there. For an (n, p) pair that is the PROTON, in either
// track order - Geant4's own comment says why: "A.R. 28-Sep-2012 Fix reproducibility problem /
// Assign the kinetic energy to the lightest of the two particles, instead to the first one
// always." So the 32 sqrt(s) values differ between pp, nn and np, and the port builds three
// grids.
//
// ## REFUSED, by name
//
//   * **the concrete channels' FINAL STATES.** `G4ConcreteNNTwoBodyResonance` inherits
//     `G4VScatteringCollision::FinalState`, which samples a resonance mass from a Breit-Wigner
//     (`SampleResonanceMass`, `BrWigInt0`, `BrWigInv`) and needs the outgoing particles' PDG
//     masses and widths. This file gives the eight PARTIAL CROSS SECTIONS and therefore the
//     channel SELECTION; what comes out of the chosen channel is the next piece.
//     `ChannelRefusal::final_state` names it at the point it would have been needed.
//   * `G4CollisionComposite::Resolve`'s **charge-balance check**, which prints to `G4cerr` and
//     changes nothing. The port asserts the same balance in the test instead, over all 306.
#ifndef G4GPU_BIC_IMR_CHANNELS_CUH
#define G4GPU_BIC_IMR_CHANNELS_CUH

#include <cmath>

#include "physics/hadronic/bic/im_r/collision_nn.cuh"
#include "physics/hadronic/bic/im_r/resonance_tables.cuh"

namespace g4gpu::bic::imr {

/// One `G4Concrete*` channel: the two entrance nucleons, the two exit particles, and which
/// column of which table its `G4XResonance` reads.
struct ConcreteChannel {
  int in1 = 0, in2 = 0;
  int out1 = 0, out2 = 0;
  int table = -1;  ///< a ResonanceTable
  int mass = 0;    ///< the resonance's nominal mass, which selects the column
  int group = -1;  ///< which of G4CollisionNN's eight components it belongs to
  int child = -1;  ///< which middle-layer child, or -1 when there is none
};

/// The PDG codes `G4HadParticleCodes.hh` declares, in the four-per-multiplet order the
/// `MakeNNTo*` templates take them (minus, zero, plus, plus-plus).
__host__ __device__ inline const int* delta_codes(int mass) {
  // Delta(1232) and the nine Delta* multiplets.
  static const int m[10] = {1232, 1600, 1620, 1700, 1900, 1905, 1910, 1920, 1930, 1950};
  static const int c[10][4] = {
      {1114, 2114, 2214, 2224},          // Delta(1232)
      {31114, 32114, 32214, 32224},      // 1600
      {1112, 1212, 2122, 2222},          // 1620
      {11114, 12114, 12214, 12224},      // 1700
      {11112, 11212, 12122, 12222},      // 1900
      {1116, 1216, 2126, 2226},          // 1905
      {21112, 21212, 22122, 22222},      // 1910
      {21114, 22114, 22214, 22224},      // 1920
      {11116, 11216, 12126, 12226},      // 1930
      {1118, 2118, 2218, 2228}};         // 1950
  for (int i = 0; i < 10; ++i) {
    if (m[i] == mass) { return c[i]; }
  }
  return nullptr;
}

/// The fifteen N* multiplets, as (proton-like, neutron-like) pairs. The first constant is named
/// `N1400pPC` in `G4HadParticleCodes.hh` and its code is the N(1440)'s; the name is a typo the
/// port does not carry, but the ORDER is the one `G4CollisionNNToNNstar`'s constructor lists.
__host__ __device__ inline const int* nstar_codes(int mass) {
  static const int m[15] = {1440, 1520, 1535, 1650, 1675, 1680, 1700, 1710,
                            1720, 1900, 1990, 2090, 2190, 2220, 2250};
  static const int c[15][2] = {
      {12212, 12112}, {2124, 1214},   {22212, 22112}, {32212, 32112}, {2216, 2116},
      {12216, 12116}, {22124, 21214}, {42212, 42112}, {32124, 31214}, {42124, 41214},
      {12218, 12118}, {52214, 52114}, {2128, 1218},   {100002210, 100002110},
      {100012210, 100012110}};
  for (int i = 0; i < 15; ++i) {
    if (m[i] == mass) { return c[i]; }
  }
  return nullptr;
}

__host__ __device__ inline const int* nstar_mass_list() {
  static const int m[15] = {1440, 1520, 1535, 1650, 1675, 1680, 1700, 1710,
                            1720, 1900, 1990, 2090, 2190, 2220, 2250};
  return m;
}
__host__ __device__ inline const int* deltastar_mass_list() {
  static const int m[9] = {1600, 1620, 1700, 1900, 1905, 1910, 1920, 1930, 1950};
  return m;
}

/// Every concrete channel `G4CollisionNN`'s constructor puts under it, generated by the same four
/// patterns `G4GeneralNNCollision` declares them with and in the same order.
///
/// The order matters twice over: `G4CollisionComposite::FinalState` accumulates partial cross
/// sections in it, and `BufferCrossSection` sums in it. Within a `MakeNNTo*` the order is the
/// GROUPn's, and across multiplets it is the constructor's call order.
inline constexpr int kConcreteChannelCount = 306;

__host__ __device__ inline int build_concrete_channels(ConcreteChannel* out, int capacity) {
  int n = 0;
  const int P = kPdgProton;
  const int N = kPdgNeutron;
  auto add = [&](int i1, int i2, int o1, int o2, int table, int mass, int group, int child) {
    if (n < capacity) {
      ConcreteChannel& c = out[n];
      c.in1 = i1; c.in2 = i2; c.out1 = o1; c.out2 = o2;
      c.table = table; c.mass = mass; c.group = group; c.child = child;
    }
    ++n;
  };
  // `MakeNNToNDelta<dm, d0, dp, dpp, channelType>::Make` - GROUP6, in its own order.
  auto make_nn_to_ndelta = [&](const int* d, int table, int mass, int group, int child) {
    const int dm = d[0], d0 = d[1], dp = d[2], dpp = d[3];
    add(N, N, N, d0, table, mass, group, child);
    add(N, N, P, dm, table, mass, group, child);
    add(N, P, P, d0, table, mass, group, child);
    add(N, P, N, dp, table, mass, group, child);
    add(P, P, N, dpp, table, mass, group, child);
    add(P, P, P, dp, table, mass, group, child);
  };
  // `MakeNNToDeltaDelta<dm, d0, dp, dpp, channelType>::Make` - GROUP10. The FIRST outgoing
  // particle is always a ground-state Delta(1232) and the second the template's multiplet.
  auto make_nn_to_deltadelta = [&](const int* d, int table, int mass, int group, int child) {
    const int* g = delta_codes(1232);
    const int Dm = g[0], D0 = g[1], Dp = g[2], Dpp = g[3];
    const int dm = d[0], d0 = d[1], dp = d[2], dpp = d[3];
    add(N, N, Dm, dp, table, mass, group, child);
    add(N, N, D0, d0, table, mass, group, child);
    add(N, N, Dp, dm, table, mass, group, child);
    add(N, P, Dp, d0, table, mass, group, child);
    add(N, P, D0, dp, table, mass, group, child);
    add(N, P, Dm, dpp, table, mass, group, child);
    add(N, P, Dpp, dm, table, mass, group, child);
    add(P, P, D0, dpp, table, mass, group, child);
    add(P, P, Dp, dp, table, mass, group, child);
    add(P, P, Dpp, d0, table, mass, group, child);
  };
  // `MakeNNToNNStar<Np, Nn, channelType>::Make` - GROUP4.
  auto make_nn_to_nnstar = [&](const int* nn, int mass, int group) {
    const int Np = nn[0], Nn = nn[1];
    add(N, N, N, Nn, kResNNstar, mass, group, -1);
    add(P, P, P, Np, kResNNstar, mass, group, -1);
    add(N, P, N, Np, kResNNstar, mass, group, -1);
    add(N, P, P, Nn, kResNNstar, mass, group, -1);
  };
  // `MakeNNToDeltaNstar<Np, channelType, Nn>::Make` - GROUP6, with a ground-state Delta out.
  auto make_nn_to_deltanstar = [&](const int* nn, int mass, int group) {
    const int* g = delta_codes(1232);
    const int Dm = g[0], D0 = g[1], Dp = g[2], Dpp = g[3];
    const int Np = nn[0], Nn = nn[1];
    add(N, N, D0, Nn, kResDeltaNstar, mass, group, -1);
    add(N, N, Dm, Np, kResDeltaNstar, mass, group, -1);
    add(P, P, Dp, Np, kResDeltaNstar, mass, group, -1);
    add(P, P, Dpp, Nn, kResDeltaNstar, mass, group, -1);
    add(N, P, D0, Np, kResDeltaNstar, mass, group, -1);
    add(N, P, Dp, Nn, kResDeltaNstar, mass, group, -1);
  };

  // component 2: G4CollisionNNToNDelta
  make_nn_to_ndelta(delta_codes(1232), kResNDelta, 1232, kNNToNDelta, -1);
  // component 3: G4CollisionNNToDeltaDelta - NOT the MakeNNToDeltaDelta template. Its own
  // constructor lists an explicit GROUP6 of Delta(1232) x Delta(1232) pairs, a different set and
  // a different order from the template its siblings use.
  {
    const int* g = delta_codes(1232);
    const int Dm = g[0], D0 = g[1], Dp = g[2], Dpp = g[3];
    add(N, N, D0, D0, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
    add(N, N, Dm, Dp, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
    add(N, P, D0, Dp, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
    add(N, P, Dm, Dpp, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
    add(P, P, Dp, Dp, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
    add(P, P, D0, Dpp, kResDeltaDelta, 1232, kNNToDeltaDelta, -1);
  }
  // component 4: G4CollisionNNToNDeltastar, nine children
  for (int i = 0; i < 9; ++i) {
    const int m = deltastar_mass_list()[i];
    make_nn_to_ndelta(delta_codes(m), kResNDeltastar, m, kNNToNDeltastar, i);
  }
  // component 5: G4CollisionNNToDeltaDeltastar, nine children
  for (int i = 0; i < 9; ++i) {
    const int m = deltastar_mass_list()[i];
    make_nn_to_deltadelta(delta_codes(m), kResDeltaDeltastar, m, kNNToDeltaDeltastar, i);
  }
  // component 6: G4CollisionNNToNNstar, fifteen multiplets, no middle layer
  for (int i = 0; i < 15; ++i) {
    const int m = nstar_mass_list()[i];
    make_nn_to_nnstar(nstar_codes(m), m, kNNToNNstar);
  }
  // component 7: G4CollisionNNToDeltaNstar, fifteen multiplets, no middle layer
  for (int i = 0; i < 15; ++i) {
    const int m = nstar_mass_list()[i];
    make_nn_to_deltanstar(nstar_codes(m), m, kNNToDeltaNstar);
  }
  return n;
}

/// `G4ConcreteNNTwoBodyResonance::IsInCharge` - the two entrance definitions against the
/// channel's own, in either order.
__host__ __device__ inline bool concrete_is_in_charge(const ConcreteChannel& c, int pdg1,
                                                      int pdg2) {
  return (pdg1 == c.in1 && pdg2 == c.in2) || (pdg1 == c.in2 && pdg2 == c.in1);
}

/// One concrete channel's cross section - `G4VCollision::CrossSection` over its `G4XResonance`.
__host__ __device__ inline double concrete_cross_section(const ConcreteChannel& c, int pdg1,
                                                         int pdg2, double sqrt_s,
                                                         ResonanceTableRefusal& ref) {
  if (!concrete_is_in_charge(c, pdg1, pdg2)) { return 0.0; }
  const int iso_out1 = resonance_iso(c.out1, ref);
  const int iso_out2 = resonance_iso(c.out2, ref);
  const int iso3_1 = (pdg1 == kPdgProton) ? 1 : -1;
  const int iso3_2 = (pdg2 == kPdgProton) ? 1 : -1;
  return x_resonance_cross_section(c.table, c.mass, 1, iso3_1, 1, iso3_2, iso_out1, iso_out2,
                                   sqrt_s, ref);
}

// =============================================================================================
// G4CollisionComposite::BufferCrossSection and G4CrossSectionBuffer.
// =============================================================================================

/// `G4CollisionComposite::nPoints`.
inline constexpr int kBufferPoints = kCompositePoints;  // 32

/// The 32 sqrt(s) values `BufferCrossSection` builds its grid at, for one pair of definitions.
///
/// `aT = theT[tt]*GeV` goes on whichever definition has the smaller PDG mass; the other stays at
/// rest. Both positions and times are zero and never read.
__host__ __device__ inline void buffer_sqrt_s_grid(double m_a, double m_b, double* out) {
  for (int tt = 0; tt < kBufferPoints; ++tt) {
    const double a_t = composite_T()[tt] * u::GeV<double>();
    double a_e = m_a;
    double b_e = m_b;
    double a_p = 0.0;
    double b_p = 0.0;
    if (m_a <= m_b) {
      a_e += a_t;
      a_p = std::sqrt(a_e * a_e - m_a * m_a);
    } else {
      b_e += a_t;
      b_p = std::sqrt(b_e * b_e - m_b * m_b);
    }
    // Both momenta are along +z, so the sum's invariant mass is built from the scalar sum.
    const double e = a_e + b_e;
    const double p = a_p + b_p;
    out[tt] = std::sqrt(e * e - p * p);
  }
}

/// `G4CrossSectionBuffer::CrossSection` - the linear search, the three index branches, the
/// backwards extrapolation below the grid, the zero above it, and the 0.01 mb floor on `y1`.
__host__ __device__ inline double buffered_cross_section(const double* grid, const double* value,
                                                         int n, double sqrt_s) {
  if (n == 1) { return value[n - 1]; }
  double x1 = 1.0, y1 = 0.0, x2 = 2.0, y2 = 0.0;
  for (int i = 0; i < n; ++i) {
    if (grid[i] > sqrt_s) {
      if (i == 0) {
        x1 = grid[0]; y1 = value[0];
        x2 = grid[1]; y2 = value[1];
      } else if (i == n - 1) {
        x1 = grid[n - 2]; y1 = value[n - 2];
        x2 = grid[n - 1]; y2 = value[n - 1];
      } else {
        x1 = grid[i - 1]; y1 = value[i - 1];
        x2 = grid[i]; y2 = value[i];
      }
      break;
    }
  }
  double result = y1 + (sqrt_s - x1) * (y2 - y1) / (x2 - x1);
  if (result < 0.0) { result = 0.0; }
  if (y1 < 0.01 * millibarn()) { result = 0.0; }
  return result;
}

/// Every buffer `G4CollisionNN`'s six resonance components need for one pair of nucleons: the
/// shared 32-point sqrt(s) grid, the nine middle-layer buffers of component 4, the nine of
/// component 5, and the six top-level ones.
///
/// This is the cascade's own state and lives in a caller-owned struct, as the brief requires -
/// Geant4 keeps it in a `std::vector<G4CrossSectionBuffer>` on the composite and grows it once
/// per particle pair seen.
struct NNChannelBuffers {
  double grid[kBufferPoints] = {};
  double ndeltastar_child[9][kBufferPoints] = {};
  double deltadeltastar_child[9][kBufferPoints] = {};
  double top[kNNChannelCount][kBufferPoints] = {};  ///< only slots 2..7 are filled
  int pdg1 = 0;
  int pdg2 = 0;
  bool built = false;
};

/// `G4CollisionComposite::BufferCrossSection` for all six resonance components at once, bottom
/// up - the children first, because a parent's node is the sum of its children's BUFFERED values
/// and each of those has already had the 0.01 mb floor applied to it.
__host__ __device__ inline void build_nn_channel_buffers(const ConcreteChannel* chans, int n_chan,
                                                         int pdg1, int pdg2, double m1, double m2,
                                                         NNChannelBuffers& buf,
                                                         ResonanceTableRefusal& ref) {
  buf.pdg1 = pdg1;
  buf.pdg2 = pdg2;
  buffer_sqrt_s_grid(m1, m2, buf.grid);
  for (int i = 0; i < 9; ++i) {
    for (int t = 0; t < kBufferPoints; ++t) {
      buf.ndeltastar_child[i][t] = 0.0;
      buf.deltadeltastar_child[i][t] = 0.0;
    }
  }
  for (int g = 0; g < kNNChannelCount; ++g) {
    for (int t = 0; t < kBufferPoints; ++t) { buf.top[g][t] = 0.0; }
  }
  // The concrete channels, summed into whichever buffer is directly above them.
  for (int c = 0; c < n_chan; ++c) {
    const ConcreteChannel& ch = chans[c];
    if (!concrete_is_in_charge(ch, pdg1, pdg2)) { continue; }
    for (int t = 0; t < kBufferPoints; ++t) {
      const double s = concrete_cross_section(ch, pdg1, pdg2, buf.grid[t], ref);
      if (ch.child >= 0) {
        if (ch.group == kNNToNDeltastar) { buf.ndeltastar_child[ch.child][t] += s; }
        else { buf.deltadeltastar_child[ch.child][t] += s; }
      } else {
        buf.top[ch.group][t] += s;
      }
    }
  }
  // The two components with a middle layer: their nodes are the sum of their children's BUFFERED
  // values at those nodes, so each child's 0.01 mb floor has already been applied.
  for (int t = 0; t < kBufferPoints; ++t) {
    double sum_nd = 0.0;
    double sum_dd = 0.0;
    for (int i = 0; i < 9; ++i) {
      sum_nd += buffered_cross_section(buf.grid, buf.ndeltastar_child[i], kBufferPoints,
                                       buf.grid[t]);
      sum_dd += buffered_cross_section(buf.grid, buf.deltadeltastar_child[i], kBufferPoints,
                                       buf.grid[t]);
    }
    buf.top[kNNToNDeltastar][t] = sum_nd;
    buf.top[kNNToDeltaDeltastar][t] = sum_dd;
  }
  buf.built = true;
}

/// The eight partial cross sections `G4CollisionComposite::FinalState` throws one uniform
/// against, in the order `G4CollisionNN`'s `GROUP8` registers them.
__host__ __device__ inline void nn_partial_cross_sections(
    int pdg1, int pdg2, const LorentzVector& p1, const LorentzVector& p2, double pdg1_mass,
    double pdg2_mass, const NNChannelBuffers& buf, double* partial_out, XsecRefusal& xref) {
  for (int i = 0; i < kNNChannelCount; ++i) { partial_out[i] = 0.0; }
  const double sqrt_s = (p1 + p2).mag();
  if (np_elastic_is_in_charge(pdg1, pdg2)) {
    partial_out[kNpElastic] =
        np_elastic_cross_section(pdg1, pdg2, p1, p2, pdg1_mass, pdg2_mass, xref);
  }
  if (nn_elastic_is_in_charge(pdg1, pdg2)) {
    partial_out[kNNElastic] =
        nn_elastic_cross_section(pdg1, pdg2, p1, p2, pdg1_mass, pdg2_mass, xref);
  }
  for (int g = kNNToNDelta; g < kNNChannelCount; ++g) {
    partial_out[g] = buffered_cross_section(buf.grid, buf.top[g], kBufferPoints, sqrt_s);
  }
}

/// `G4CollisionComposite::FinalState`'s SELECTION - the running sum against one uniform, and the
/// index of the component it lands in. Returns -1 for Geant4's `return NULL`, which happens when
/// every partial is zero (so `partialCxSum` is 0, `random` is 0, and `running > random` is never
/// true) or when rounding leaves the uniform above the sum.
template <typename Rng>
__host__ __device__ inline int nn_select_channel(const double* partial, Rng& rng) {
  double partial_sum = 0.0;
  for (int i = 0; i < kNNChannelCount; ++i) { partial_sum += partial[i]; }
  const double random = rng.uniform() * partial_sum;
  double running = 0.0;
  for (int i = 0; i < kNNChannelCount; ++i) {
    running += partial[i];
    if (running > random) { return i; }
  }
  return -1;
}

}  // namespace g4gpu::bic::imr

#endif
