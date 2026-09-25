// Philox4x32-10 counter-based RNG. Keyed by (event, track, purpose) so a track's
// stream is independent of scheduling order and reproducible run to run.
#pragma once
#include <cstdint>
namespace g4gpu {

__host__ __device__ inline uint32_t mulhi32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

/// 32-bit avalanche finalizer (the "lowbias32" constants). Used to derive a deterministic,
/// well-separated RNG key for a secondary from its parent, so a track's stream never depends
/// on nondeterministic scheduling. See docs/RISK.md, reproducibility.
__host__ __device__ inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

/// Deterministic RNG key for a secondary. Distinct parents, distinct steps, and distinct
/// child slots within one step all yield uncorrelated keys. Key collisions are possible in
/// principle (32 bits, ~40 tracks per event -> ~2e-7) and would only mean two tracks share a
/// stream, not a failure.
__host__ __device__ inline uint32_t child_rng_key(uint32_t parent_key, uint32_t parent_step,
                                                  uint32_t child_index) {
  return mix32(parent_key ^ mix32(parent_step * 0x9E3779B9u + child_index + 1u));
}

template <typename real_t>
class Philox {
 public:
  __host__ __device__ Philox(uint32_t event, uint32_t track, uint32_t purpose = 0)
      : ctr_{0u, event, track, purpose}, key_{0xA341316Cu, 0xC8013EA4u}, idx_(4) {}

  /// Uniform in (0,1). Never returns exactly 0 or 1, so log() and division are safe.
  __host__ __device__ real_t uniform() {
    if (idx_ >= 4) { advance(); idx_ = 0; }
    return uniform_from_word(buf_[idx_++]);
  }

  /// The word-to-(0,1) map, a static so a test can hand it the two extreme words. In double,
  /// `(r + 0.5) * 2^-32` is exact and lies in [2^-33, 1 - 2^-33]. In float the same expression
  /// ROUNDS: a word above 2^32 - 2^8 converts to 2^32, the half is lost, and the product is
  /// exactly 1.0f - which RandGaussQ turns into a NaN (docs/RISK.md V186) - so the float map
  /// keeps the top 23 bits, so that the half-offset sum has 24 significant bits, which is what a
  /// float carries exactly (24 bits plus the half is 25, and rounds back up to 1.0f - measured):
  /// `((r >> 9) + 0.5) * 2^-23` lies in [2^-24, 1 - 2^-24]. The double map is unchanged, and
  /// every recorded stream in this repository depends on it staying so.
  __host__ __device__ static real_t uniform_from_word(uint32_t r) {
    if constexpr (sizeof(real_t) == 4) {
      return (real_t(r >> 9) + real_t(0.5)) * real_t(1.1920928955078125e-07);  // /2^23
    } else {
      return (real_t(r) + real_t(0.5)) * real_t(2.3283064365386963e-10);  // /2^32
    }
  }

 private:
  __host__ __device__ void advance() {
    uint32_t c[4] = {ctr_[0], ctr_[1], ctr_[2], ctr_[3]};
    uint32_t k[2] = {key_[0], key_[1]};
    for (int r = 0; r < 10; ++r) {
      const uint32_t hi0 = mulhi32(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
      const uint32_t hi1 = mulhi32(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
      c[0] = hi1 ^ c[1] ^ k[0];
      c[1] = lo1;
      c[2] = hi0 ^ c[3] ^ k[1];
      c[3] = lo0;
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
    for (int i = 0; i < 4; ++i) { buf_[i] = c[i]; }
    ++ctr_[0];  // next block in this track's stream
  }

  uint32_t ctr_[4];
  uint32_t key_[2];
  uint32_t buf_[4]{};
  int idx_;
};

}  // namespace g4gpu
