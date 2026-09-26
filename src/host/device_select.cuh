// Which GPU the engine and the drivers run on, chosen once from G4GPU_DEVICE.
//
// Every driver and the engine took device 0, which is right for one card and wrong the day a
// second one is in the machine - the Titan V (compute capability 7.0) beside the RTX 3070 -
// because Windows numbers the cards in an order nobody chose. `G4GPU_DEVICE` names the card by
// index when the number is one of the devices present, otherwise by a substring of its name,
// case-insensitively ("titan", "3070"); unset, it is device 0 as before, so a one-card machine
// sees no change. The choice is made once, on the first call, and printed, because a run that
// silently landed on the wrong card would be a mystery in every measurement after it; a name or
// index that matches nothing is fatal for the same reason.
// `cudaSetDevice` must precede the first allocation on the context, which is why `Upload` and
// every driver call this before anything else touches CUDA.
#pragma once

#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace g4gpu::host {

struct SelectedDevice {
  int index = 0;          ///< the device `cudaSetDevice` was given
  int count = 0;          ///< how many CUDA devices the runtime reports
  cudaDeviceProp prop{};  ///< its properties, valid when `ok`
  bool ok = false;        ///< false when there is no device at all
};

/// Case-insensitive "needle occurs in haystack", for matching G4GPU_DEVICE against a name.
inline bool device_name_contains(const char* haystack, const char* needle) {
  const std::size_t n = std::strlen(needle);
  if (n == 0) { return true; }
  for (const char* h = haystack; *h != '\0'; ++h) {
    std::size_t i = 0;
    while (i < n && h[i] != '\0' &&
           std::tolower(static_cast<unsigned char>(h[i])) ==
               std::tolower(static_cast<unsigned char>(needle[i]))) {
      ++i;
    }
    if (i == n) { return true; }
  }
  return false;
}

/// The device this process uses, selected on the first call and the same ever after.
inline const SelectedDevice& select_device() {
  static const SelectedDevice chosen = [] {
    SelectedDevice d;
    if (cudaGetDeviceCount(&d.count) != cudaSuccess || d.count <= 0) {
      d.count = 0;
      return d;
    }
    int pick = 0;
    const char* want = std::getenv("G4GPU_DEVICE");
    if (want != nullptr && *want != '\0') {
      // A number that is a valid index is an index; anything else, "3070" included, is matched
      // against the names - a card's name is the natural way to ask for it, and a bare number
      // beyond the count is not an index anyone meant.
      char* end = nullptr;
      const long idx = std::strtol(want, &end, 10);
      pick = -1;
      if (end != want && *end == '\0' && idx >= 0 && idx < d.count) {
        pick = static_cast<int>(idx);
      } else {
        for (int i = 0; i < d.count; ++i) {
          cudaDeviceProp p{};
          if (cudaGetDeviceProperties(&p, i) == cudaSuccess && device_name_contains(p.name, want)) {
            pick = i;
            break;
          }
        }
      }
      if (pick < 0) {
        std::fprintf(stderr, "FATAL: G4GPU_DEVICE=%s names none of the %d CUDA devices:\n", want,
                     d.count);
        for (int i = 0; i < d.count; ++i) {
          cudaDeviceProp p{};
          if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
            std::fprintf(stderr, "  %d: %s, CC %d.%d\n", i, p.name, p.major, p.minor);
          }
        }
        std::exit(2);
      }
    }
    if (cudaSetDevice(pick) != cudaSuccess) {
      std::fprintf(stderr, "FATAL: cudaSetDevice(%d) failed: %s\n", pick,
                   cudaGetErrorString(cudaGetLastError()));
      std::exit(2);
    }
    d.index = pick;
    d.ok = (cudaGetDeviceProperties(&d.prop, pick) == cudaSuccess);
    if (d.ok) {
      std::printf("GPU: %s (device %d of %d), CC %d.%d\n", d.prop.name, d.index, d.count,
                  d.prop.major, d.prop.minor);
    }
    return d;
  }();
  return chosen;
}

}  // namespace g4gpu::host
