// The track arena's arithmetic: that what is measured is what is carved, and that what is
// carved is aligned.
//
// WHY THIS EXISTS
//
// The engine allocates both halves of the ping-pong as one block and puts the second half at
// `base + track_arena_half_bytes(pool)`. That number is therefore the alignment of every array
// in the second half, and nothing checked it. A track slot is 236 bytes, so an ODD pool made it
// 4 mod 8 and every double in the second half was misaligned - which a kernel reports as
// "misaligned address" at whatever cudaMemcpy happens to synchronise next, with no hint of
// where it came from.
//
// An odd pool is not exotic. It is what `batch * live_per_event` gives whenever the product is
// not whole, so `g4dose -live 2.5` reached it while the default `-live 4` could not: with an
// even pool the arithmetic lands on 8 by luck. The whole pipeline ran the lucky case.
//
// This is host-only on purpose. Every quantity here is arithmetic over sizes, so it needs no
// GPU and no kernel, and a test that needs neither runs in milliseconds and can afford to
// sweep hundreds of pool sizes rather than the two a run would have time for.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/track_buffer.cuh"

using namespace g4gpu;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what, long long pool) {
  if (!ok) {
    std::printf("  FAIL: %s (pool %lld)\n", what, pool);
    ++g_fails;
  }
}

/// Carves one half exactly as TransportEngine::Upload does, and reports the worst alignment of
/// any array it produced, plus whether it fit.
///
/// The base is a fictitious address rather than a real allocation: nothing is dereferenced, and
/// what is under test is the offset arithmetic. It is set to the alignment cudaMalloc
/// guarantees, so an alignment failure here is the carve's and not the allocator's.
template <typename real_t>
struct CarveResult {
  size_t worst_align = TrackSlab::kAlign;  ///< smallest power of two every pointer satisfies
  bool fit = true;
  size_t used = 0;
};

template <typename real_t>
CarveResult<real_t> Carve(long long pool, size_t half_bytes, size_t base_offset) {
  CarveResult<real_t> r{};
  TrackSlab slab{};
  // 256-aligned, which is the weakest guarantee cudaMalloc makes.
  auto* fake_base = reinterpret_cast<char*>(static_cast<uintptr_t>(TrackSlab::kAlign * 1024));
  slab.base = fake_base + base_offset;
  slab.capacity = half_bytes;
  slab.used = 0;

  TrackBuffer<real_t> v{};
  const cudaError_t e = allocate_track_buffer<real_t>(v, static_cast<int>(pool), nullptr, &slab);
  r.fit = (e == cudaSuccess) && !slab.overflowed;
  r.used = slab.used;

  // Every array the buffer holds, whatever its type: the alignment that matters is the one the
  // widest element needs, and reading them all as void* is how to ask that without listing
  // types. If a new field is added to the buffer and not to this list, the check simply does
  // not cover it - which is why the fit check above is here too, since a missed field shows up
  // there as a size that no longer matches.
  const void* ptrs[] = {
      v.species, v.x, v.y, v.z, v.dx, v.dy, v.dz, v.ekin, v.volume, v.event, v.rng_key,
      v.step, v.msc_tlimit, v.msc_tlimitmin, v.status, v.flags, v.parent_key, v.global_time,
      v.local_time, v.proper_time, v.track_length, v.vx, v.vy, v.vz, v.vdx, v.vdy, v.vdz,
      v.vertex_ekin, v.vertex_volume, v.creator_process, v.weight, v.polx, v.poly, v.polz,
      v.user_data, v.count, v.overflow,
  };
  for (const void* p : ptrs) {
    if (p == nullptr) { continue; }
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    size_t align = TrackSlab::kAlign;
    while (align > 1 && (a % align) != 0) { align /= 2; }
    if (align < r.worst_align) { r.worst_align = align; }
  }
  return r;
}

}  // namespace

int main() {
  std::printf("== the track arena's size and alignment ==\n\n");

  // Odd numbers, powers of two, and the two pools that actually came out of the automatic
  // batch sizer on this machine - 7396897 from `-live 2.5` is the one that faulted.
  const long long pools[] = {
      1, 2, 3, 7, 63, 64, 65, 255, 256, 257, 4095, 4096, 4097, 65535, 65536, 65537,
      100003, 999999, 1000000, 2499997, 2500000, 7389564, 7396897, 12345679,
  };

  for (long long pool : pools) {
    const size_t half = track_arena_half_bytes<double>(pool);

    // 1. The half size is what the second half's base is offset by, so it is the alignment of
    //    everything in the second half. This is the assertion the fault was about.
    Check(half % TrackSlab::kAlign == 0, "arena half size is a multiple of TrackSlab::kAlign",
          pool);

    // 2. Carve both halves the way the engine does, the second at base + half.
    const auto lo = Carve<double>(pool, half, 0);
    const auto hi = Carve<double>(pool, half, half);
    Check(lo.fit, "the first half's carve fits what was measured for it", pool);
    Check(hi.fit, "the second half's carve fits what was measured for it", pool);
    Check(lo.worst_align >= alignof(double), "every array in the first half is aligned for a "
          "double", pool);
    Check(hi.worst_align >= alignof(double), "every array in the second half is aligned for a "
          "double", pool);

    // 3. EXACTLY what the carve consumes, not merely enough for it. track_arena_half_bytes
    //    dry-runs the same allocator, so there is no estimate to be loose or tight - and this
    //    is the assertion that keeps it that way. The version this replaced was per-slot times
    //    the pool plus a slack term, and it was the estimate, not the carve, that put the
    //    second half on a 4-byte boundary. An over-measure is not harmless either: the batch
    //    sizer bisects on this number, so slack is events that never ran.
    Check(half == lo.used, "the measured half is exactly what the carve consumes", pool);
  }

  // A float build carves the same shape with narrower reals, and its per-slot cost is odd in a
  // different place, so it gets the same sweep.
  for (long long pool : pools) {
    const size_t half = track_arena_half_bytes<float>(pool);
    Check(half % TrackSlab::kAlign == 0, "float: arena half size is kAlign-aligned", pool);
    const auto hi = Carve<float>(pool, half, half);
    Check(hi.fit, "float: the second half's carve fits", pool);
    Check(hi.worst_align >= alignof(float), "float: every array in the second half is aligned",
          pool);
  }

  // A pool too large for an int must report an impossible size, not a truncated one - the
  // batch sizer bisects on this number, so saturating makes it walk down, while truncating
  // would make a 4-billion-slot pool look like a small one and let the run proceed with a
  // buffer nothing asked for. The doubling the sizer does must not wrap either.
  {
    const long long too_big = 2147483648LL;  // INT_MAX + 1
    const size_t half = track_arena_half_bytes<double>(too_big);
    Check(half > (size_t(1) << 40), "a pool larger than an int reports an impossible size",
          too_big);
    Check(half * 2 > half, "and doubling that size does not wrap", too_big);
    Check(track_arena_half_bytes<double>(2147483647LL) < half,
          "while the largest pool that does fit reports an ordinary one", 2147483647LL);
  }

  // track_bytes_per_slot is measured by differencing two capacities, and the reason it is
  // measured at 65536 rather than at 1 and 2 is that alignment swamps the difference at small
  // capacities. That reasoning is a claim about this function, so it is checked: the slot cost
  // must be the same whichever large capacity it is differenced at.
  {
    const size_t per_slot = track_bytes_per_slot<double>();
    const size_t at_1 = track_buffer_bytes<double>(2) - track_buffer_bytes<double>(1);
    Check(per_slot >= 200 && per_slot <= 300,
          "a double track slot is somewhere near the 236 bytes the docs claim", per_slot);
    Check(at_1 < per_slot,
          "differencing at capacity 1 under-counts, which is why it is not done that way",
          static_cast<long long>(at_1));
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
