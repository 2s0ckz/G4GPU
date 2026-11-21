// P41: the de-excitation tail OFF the interaction slots - an entry of a wide tail pool holds
// what one flat cascade's tail touches, sized from what the tail was measured to touch, instead
// of the 1.6 MB thread-path `InteractionSlot` P36's tail borrowed (docs/RISK.md V325-V327).
//
// WHAT A TAIL TOUCHES (V325's tape, the proton and Fe56 beams). The tail of a Binary cascade or a
// light-ion reaction is `Propagate` after `FindFragments` - the residual's PreCompound and
// G4ExcitationHandler (`preco::deexcite`, which calls `deex::deexcite`), the final decays, the
// product lists - then `ApplyYourself`'s packing, `CheckResult`, the K0 mixing and the emission.
// On the slot that is the de-excitation's 536,576 B (evaporation list 4,096 fragments, results
// 1,024, step 512, products 1,024, PreCompound's list 1,024), the cascade's 512 products and 256
// PreCompound products, `BicScratch`'s 22,528-B decay list, `bic_fs`/`blir_fs` and three
// 856-wide final-state buffers (`fs`, `filled`, `pdg_mass`: 143 kB). V680 (P38) and V325 measured
// what is written of it: the handler's lists at most 5 / 27 / 26 / 27 / 27 entries, the products
// at most 26 (Binary) and 85 (light ion). An entry here is those lists at a margin over the
// measured maxima.
//
// ONE FORMULA, TWO PLACES. The tail's code is the slot tail's code: `bic::flat::stage_tail`,
// `flat_tail_precompound`, `flat_pack_nucleon`, `ion_pack` take their buffers as views, and the
// views below point them at an entry instead of a slot. `preco::deexcite` and `deex::deexcite`
// are called unchanged; only the de-excitation workspace they are handed is smaller, and when it
// binds (`refused_capacity`, `ref.capacity`) the call is CONTINUED - the generator put back where
// it was, the same call on a slot-sized workspace borrowed from a small pool - never rerun from
// the key (a rerun is a different cascade, and the rerun subset is selected by the failure: the
// lead's note of 2026-11-20, from P39's measurement). So the workspace's size moves time, never a
// number.
//
// THE AREA. The de-excitation's lists, the decay list and the copied final state are never live
// at once: the decay pass starts after `bic_deexcite_fragment` has copied the handler's products
// out, and the final state is copied after the packing has read the decay pass's products. So
// they share one area, and an entry is the largest of the three plus the lists that do overlap
// (the PreCompound products, the cascade products, the model's own final state). Nothing in the
// area is read before it is written in its phase: the handler's lists are append-only
// (`FragmentList::push` assigns whole elements), the decay list is assigned element by element
// before `decay_kinetic_tracks`, and `copy_final_state` starts with `clear()`.
#ifndef G4GPU_DEEX_FLAT_TAIL_POOL_CUH
#define G4GPU_DEEX_FLAT_TAIL_POOL_CUH

#include <cstddef>

#include "physics/hadronic/bic/flat/ion_flat.cuh"

namespace g4gpu::deex::flat {

/// One pool's capacities. The handler's five (evaporation list, results, one BreakFragment's
/// step, its products, PreCompound's list), the cascade's PreCompound products and products,
/// the decay list, the light-ion arm's two `SortResult` lists, and the copied final state.
template <int kEvap_, int kResults_, int kStep_, int kDeexOut_, int kPrecoOut_, int kPreco_,
          int kProducts_, int kDecay_, int kSpect_, int kCasc_, int kFs_>
struct TailCaps {
  static constexpr int kEvap = kEvap_;
  static constexpr int kResults = kResults_;
  static constexpr int kStep = kStep_;
  static constexpr int kDeexOut = kDeexOut_;
  static constexpr int kPrecoOut = kPrecoOut_;
  static constexpr int kPreco = kPreco_;
  static constexpr int kProducts = kProducts_;
  static constexpr int kDecay = kDecay_;
  static constexpr int kSpect = kSpect_;
  static constexpr int kCasc = kCasc_;
  static constexpr int kFs = kFs_;
};

/// The two pools' capacities. ONLY the de-excitation workspace is compact - over the measured
/// maxima (V680/V325: the handler's lists 5 / 27 / 26 / 27 / 27) - because only its refusal can be
/// continued in place (the generator put back, the same call on a slot-sized workspace,
/// `bic::flat::TailFallback`): every list the tail writes AFTER the record has moved on (the
/// PreCompound and cascade products, the decay list, `SortResult`'s two) is at the slot's own
/// capacity, so it refuses exactly when the slot tail would. The final state is the model's own
/// (`kBicMaxSecondaries` 128, `kBlirMaxSecondaries` 168), so `copy_final_state` never counts an
/// overflow the slot's 856 would not.
using BinaryTailCaps = TailCaps<16, 64, 64, 64, 64, bic::kSlotPrecoCap, bic::kSlotProductCap,
                                bic::kCascadeMaxScratch, 1, 1, bic::kBicMaxSecondaries>;
using IonTailCaps =
    TailCaps<16, 64, 64, 64, 64, bic::kSlotPrecoCap, bic::kSlotProductCap, bic::kCascadeMaxScratch,
             bic::kSlotSpectatorCap, bic::kSlotCascaderCap, bic::kBlirMaxSecondaries>;

/// The area's three tenants, one at a time.
template <class Caps>
struct DeexArea {
  Fragment evap[Caps::kEvap];
  Fragment results[Caps::kResults];
  Fragment step[Caps::kStep];
  DeexProduct deex_out[Caps::kDeexOut];
  DeexProduct preco_out[Caps::kPrecoOut];
};
template <class Caps>
struct DecayArea {
  bic::DecayTrack list[Caps::kDecay];
};
template <class Caps>
struct FsArea {
  physics::hadronic::HadFinalState<double, Caps::kFs> fs;
  double pdg_mass[Caps::kFs];
};

constexpr std::size_t max3(std::size_t a, std::size_t b, std::size_t c) {
  return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

/// What the tail hands the emission (`flat_tail_emit`): nothing to do, a final state in
/// `fs_area`, or an entry the integral rejection answered without a model call.
enum TailReady : int { kTailNotReady = 0, kTailFinalState = 1, kTailRejected = 2 };

/// One entry: the area, the lists that overlap it, the model's final state, and the hand-over.
/// `ModelFs` is `bic::BicFinalState` or `bic::BlirFinalState`, the type `flat_pack_nucleon` and
/// `ion_pack` fill.
template <class Caps, class ModelFs>
struct TailEntry {
  using caps = Caps;
  static constexpr std::size_t kArea =
      max3(sizeof(DeexArea<Caps>), sizeof(DecayArea<Caps>), sizeof(FsArea<Caps>));
  alignas(16) unsigned char area[kArea];
  bic::CascadeProduct preco_products[Caps::kPreco];
  bic::CascadeProduct products[Caps::kProducts];
  bic::BlirProduct spectators[Caps::kSpect];
  bic::BlirProduct cascaders[Caps::kCasc];
  bic::BlirProduct cascaders_save[Caps::kCasc];   ///< the spectators' continuation (`ion_pack`)
  ModelFs model_fs;
  int lane;
  int ready;
};

/// The handler's and PreCompound's workspace, on the entry's area.
template <class E>
__host__ __device__ inline preco::PrecoWorkspace tail_deex_workspace(E& e) {
  using C = typename E::caps;
  DeexArea<C>* d = reinterpret_cast<DeexArea<C>*>(e.area);
  preco::PrecoWorkspace w;
  w.deex.evap_list = d->evap;
  w.deex.evap_capacity = C::kEvap;
  w.deex.results = d->results;
  w.deex.results_capacity = C::kResults;
  w.deex.step = d->step;
  w.deex.step_capacity = C::kStep;
  w.deex.products = d->deex_out;
  w.deex.products_capacity = C::kDeexOut;
  w.products = d->preco_out;
  w.products_capacity = C::kPrecoOut;
  return w;
}

/// `G4DecayKineticTracks`' list, on the area (after the de-excitation).
template <class E>
__host__ __device__ inline bic::DecayTrack* tail_decay_list(E& e) {
  return reinterpret_cast<DecayArea<typename E::caps>*>(e.area)->list;
}

/// The copied final state and its definition masses, on the area (after the packing).
template <class E>
__host__ __device__ inline FsArea<typename E::caps>& tail_fs(E& e) {
  return *reinterpret_cast<FsArea<typename E::caps>*>(e.area);
}

/// The `CascadeWorkspace` `stage_tail` and `flat_pack_nucleon` read: the two product lists. No
/// track pool, collision list or `BicScratch` - the tail reads none of them (the one
/// `DecayVoidNucleus` that would is sent to a slot, `tail_needs_slot`).
template <class E>
__host__ __device__ inline bic::CascadeWorkspace tail_cascade_view(E& e) {
  using C = typename E::caps;
  bic::CascadeWorkspace w;
  w.products = e.products;
  w.product_capacity = C::kProducts;
  w.preco_products = e.preco_products;
  w.preco_capacity = C::kPreco;
  w.scratch = nullptr;
  return w;
}

/// The `BlirStorage` `ion_pack` reads: the products and `SortResult`'s two lists.
template <class E>
__host__ __device__ inline bic::BlirStorage tail_blir_view(E& e) {
  using C = typename E::caps;
  bic::BlirStorage s;
  s.cascade = tail_cascade_view(e);
  s.spectators = e.spectators;
  s.spectator_capacity = C::kSpect;
  s.cascaders = e.cascaders;
  s.cascader_capacity = C::kCasc;
  return s;
}

/// Which tails need a thread-path slot rather than an entry: `apply_yourself_h1` (A == 1: a
/// nucleus build and `Propagate1H1` on the slot's cascade arrays - no de-excitation at all) and
/// `DecayVoidNucleus` (an all-neutron remnant: its lists are `BicScratch`'s). Both are a function
/// of the record alone.
template <int N, int P, int C>
__host__ __device__ inline bool tail_needs_slot(const bic::flat::FlatCascade<N, P, C>& c) {
  return c.tail_work == bic::flat::kTailHydrogen ||
         (c.tail_work == bic::flat::kTailDeexcite && c.frag.a == 0);
}

/// The continuation's workspaces (`bic::flat::TailFallback`) on the device: `n` at the slot's
/// capacities (4,096 / 1,024 / 512 fragments, 1,024 / 1,024 products: 536,576 B each), their
// views and a flag each; `busy[n]` counts the continuations. `bytes` is 0 when they could not be
/// allocated (then nothing is held).
struct BigWorkspaces {
  void* arrays = nullptr;
  preco::PrecoWorkspace* views = nullptr;
  int* busy = nullptr;
  int n = 0;
  std::size_t bytes = 0;
};
inline BigWorkspaces build_big_workspaces(int n) {
  BigWorkspaces b;
  constexpr int kE = 4096, kR = 1024, kS = 512, kD = 1024, kQ = 1024;
  const std::size_t one =
      sizeof(Fragment) * (kE + kR + kS) + sizeof(DeexProduct) * (kD + kQ);
  if (n <= 0 || cudaMalloc(&b.arrays, one * n) != cudaSuccess ||
      cudaMalloc(&b.views, sizeof(preco::PrecoWorkspace) * n) != cudaSuccess ||
      cudaMalloc(&b.busy, sizeof(int) * (n + 1)) != cudaSuccess) {
    cudaGetLastError();
    cudaFree(b.arrays);
    cudaFree(b.views);
    cudaFree(b.busy);
    return BigWorkspaces{};
  }
  preco::PrecoWorkspace* v = new preco::PrecoWorkspace[n];
  for (int k = 0; k < n; ++k) {
    unsigned char* p = static_cast<unsigned char*>(b.arrays) + one * k;
    Fragment* fr = reinterpret_cast<Fragment*>(p);
    DeexProduct* dp = reinterpret_cast<DeexProduct*>(p + sizeof(Fragment) * (kE + kR + kS));
    v[k].deex.evap_list = fr;
    v[k].deex.evap_capacity = kE;
    v[k].deex.results = fr + kE;
    v[k].deex.results_capacity = kR;
    v[k].deex.step = fr + kE + kR;
    v[k].deex.step_capacity = kS;
    v[k].deex.products = dp;
    v[k].deex.products_capacity = kD;
    v[k].products = dp + kD;
    v[k].products_capacity = kQ;
  }
  cudaMemcpy(b.views, v, sizeof(preco::PrecoWorkspace) * n, cudaMemcpyHostToDevice);
  delete[] v;
  cudaMemset(b.busy, 0, sizeof(int) * (n + 1));
  b.n = n;
  b.bytes = one * n + (sizeof(preco::PrecoWorkspace) + sizeof(int)) * n + sizeof(int);
  return b;
}
inline bic::flat::TailFallback fallback_of(const BigWorkspaces& b) {
  bic::flat::TailFallback f;
  f.big = b.views;
  f.busy = b.busy;
  f.n = b.n;
  return f;
}

/// The cost bin (0 .. 15) a waiting tail is sorted into before its launch, from what the record
/// knows before the tail runs. PROVISIONAL until V325's tape says what predicts the time.
template <int N, int P, int C>
__host__ __device__ inline int tail_cost_bin(const bic::flat::FlatCascade<N, P, C>& c) {
  double key = 0.0;
  if (c.tail_work == bic::flat::kTailDeexcite && c.frag.a > 1) {
    key = c.excitation;
  } else if (c.tail_work == bic::flat::kTailPrecompound) {
    key = c.kin;
  } else {
    return 0;
  }
  const int b = 1 + static_cast<int>(key / 10.0);
  return (b > 15) ? 15 : ((b < 1) ? 1 : b);
}

}  // namespace g4gpu::deex::flat

#endif
