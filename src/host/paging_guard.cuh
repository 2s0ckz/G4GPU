// A device allocation the card cannot hold is REFUSED, by name, before it is made.
//
// WHY. CUDA on Windows (WDDM) does not fail an allocation that is larger than the free device
// memory: it oversubscribes into system memory and the run pages over PCIe. docs/RISK.md V239
// measured what that costs - the Titan V's proton depth-dose gate took 48 minutes cold and about
// an hour and three quarters inside the gate where the 3070 takes seconds, with the device stack
// raise reporting "0.00 GB of 12.00 GB left" and the card reporting 12.6 GB in use - and
// 2026-11-16 a 200,000-event proton beam at 32 live tracks an event drove the host process to
// 26.7 GB of working set before it was stopped by hand. The batch had been refused this way
// since the track buffers first did it (`Upload`'s "Refused rather than attempted"); the device
// stack raise, the interaction pool and the two interaction queues were not, and those are what
// paged. docs/RISK.md V258.
//
// THE RULE. Each of those allocations reads `cudaMemGetInfo` immediately before it is made and
// refuses, with a FATAL line naming what was asked, what is free, the card, and - for the stack -
// the resident-thread count and the frame, when the request is larger than the free memory.
// `G4GPU_ALLOW_PAGING=1` keeps the old behaviour for a deliberate measurement of the paging
// itself, and says that it is on every time it lets a request through that would page.
//
// The decision is `paging_verdict`, a pure function of the numbers, so that a host test can
// assert it without a card that is short of memory; `refuse_if_paging` is the one place that
// prints and exits.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace g4gpu::host {

/// What one allocation asks, against what the card has.
struct PagingRequest {
  const char* what = "";          ///< a name for the allocation, as the FATAL line prints it
  std::size_t asked = 0;          ///< bytes the allocation will take from the card
  std::size_t free_bytes = 0;     ///< `cudaMemGetInfo`'s free, read just before
  std::size_t total_bytes = 0;    ///< and its total
  const char* card = "";          ///< `cudaDeviceProp::name`
  long long resident_threads = 0; ///< the stack raise only: SMs x max threads an SM, else 0
  std::size_t frame = 0;          ///< the stack raise only: the largest kernel frame, else 0
  std::size_t per_thread = 0;     ///< the stack raise only: the bytes a thread reserved
};

enum class PagingVerdict : int {
  kFits = 0,      ///< the card holds it
  kRefused = 1,   ///< larger than the free memory, and paging is not allowed
  kAllowed = 2    ///< larger than the free memory, and G4GPU_ALLOW_PAGING=1 lets it through
};

/// `G4GPU_ALLOW_PAGING=1` - the one switch that lets a request the card cannot hold go ahead.
inline bool paging_allowed_by_env() {
  const char* v = std::getenv("G4GPU_ALLOW_PAGING");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

/// The decision, and nothing else. A request equal to the free memory fits.
inline PagingVerdict paging_verdict(std::size_t asked, std::size_t free_bytes, bool allow) {
  if (asked <= free_bytes) { return PagingVerdict::kFits; }
  return allow ? PagingVerdict::kAllowed : PagingVerdict::kRefused;
}

/// Prints and exits (status 2) on `kRefused`, prints a notice on `kAllowed`, and is silent when
/// the request fits. Returns the verdict for the caller that wants to report it.
inline PagingVerdict refuse_if_paging(const PagingRequest& r) {
  const PagingVerdict v = paging_verdict(r.asked, r.free_bytes, paging_allowed_by_env());
  if (v == PagingVerdict::kFits) { return v; }
  const double gb = 1073741824.0;
  if (v == PagingVerdict::kAllowed) {
    std::printf("G4GPU_ALLOW_PAGING=1: %s asks %.3f GB with %.3f GB of %.2f GB free on %s - "
                "let through, and WDDM will back the difference with system memory "
                "(docs/RISK.md V258)\n",
                r.what, double(r.asked) / gb, double(r.free_bytes) / gb,
                double(r.total_bytes) / gb, r.card);
    return v;
  }
  std::printf("\nFATAL: %s asks %zu bytes (%.3f GB) of the device and %zu bytes (%.3f GB) of\n"
              "       %.2f GB are free on %s.\n",
              r.what, r.asked, double(r.asked) / gb, r.free_bytes, double(r.free_bytes) / gb,
              double(r.total_bytes) / gb, r.card);
  if (r.resident_threads > 0) {
    std::printf("       The device stack is reserved for every thread the card can hold: %lld\n"
                "       resident threads x %zu bytes a thread, off a largest kernel frame of\n"
                "       %zu bytes.\n",
                r.resident_threads, r.per_thread, r.frame);
  }
  std::printf("       Refused rather than attempted: CUDA on Windows backs an allocation the\n"
              "       card cannot hold with system memory, and the run pages instead of\n"
              "       failing. Lower the batch or SetInteractionSlots, or set\n"
              "       G4GPU_ALLOW_PAGING=1 to measure the paging deliberately.\n"
              "       docs/RISK.md V239, V258.\n");
  std::exit(2);
}

}  // namespace g4gpu::host
