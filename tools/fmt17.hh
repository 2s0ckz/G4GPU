// A round-trippable double formatter whose output does not depend on the C runtime.
//
// `printf("%.17g", 1.09e-22)` gives `1.0900000000000001e-22` under MSVC and
// `1.0900000000000001e-022` under the MinGW g++ that ships with Git for Windows. Same value,
// same mantissa, different exponent width - C requires *at least* two exponent digits and says
// nothing about a maximum, so both are conforming.
//
// That is harmless in a program and corrosive in a *generated file that is checked in*.
// src/data/barashenkov.hh was generated with nvcc and re-generated with g++, and 582 lines
// changed without a single value changing. The whole point of tools/refresh_tables.sh is to
// answer "did anything move when I upgraded Geant4", and an answer that depends on which
// compiler built the generator is not an answer.
//
// So: format with %.17g, then trim the exponent to the two-digit minimum. Every generator
// under tools/ that writes a double into src/data goes through this.
#ifndef G4GPU_TOOLS_FMT17_HH
#define G4GPU_TOOLS_FMT17_HH

#include <cstdio>
#include <cstring>

namespace g4gpu::tools {

/// Formats @p x with %.17g into @p buf, then normalises the exponent to two digits.
///
/// Returns @p buf, so it can be used inline in a printf argument list - but note that a caller
/// printing two values in one call needs two buffers, which is why this does not return a
/// static one.
inline const char* fmt17(double x, char* buf, std::size_t n) {
  std::snprintf(buf, n, "%.17g", x);
  char* e = std::strchr(buf, 'e');
  if (e == nullptr) { e = std::strchr(buf, 'E'); }
  if (e == nullptr) { return buf; }  // no exponent, nothing to normalise

  char* p = e + 1;
  if (*p == '+' || *p == '-') { ++p; }  // step over the sign

  // Count the digits, then drop leading zeros while more than two remain. Two is C's floor,
  // so this is the narrowest form every conforming runtime can also produce.
  std::size_t digits = std::strlen(p);
  std::size_t strip = 0;
  while (strip < digits && p[strip] == '0' && digits - strip > 2) { ++strip; }
  if (strip > 0) { std::memmove(p, p + strip, digits - strip + 1); }
  return buf;
}

}  // namespace g4gpu::tools

#endif  // G4GPU_TOOLS_FMT17_HH
