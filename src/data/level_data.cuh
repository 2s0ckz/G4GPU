// Nuclear level data: the PhotonEvaporation5.7 reader and the flat tables it fills.
//
// Transcribed from G4LevelReader::LevelManager, G4LevelManager and G4NucLevel (11.1.1). Host
// reader into flat arrays, device-callable accessors over them - the pattern
// src/data/photoelectric_data.cuh established.
//
// The file format, one nuclide per file `z<Z>.a<A>`, whitespace-separated:
//
//   <level index> <floating> <E keV> <lifetime> <spin> <ntrans>
//       <final index> <E_trans keV> <relative intensity> <multipolarity> <mp ratio> <alpha>
//       [ten internal-conversion coefficients, present only when alpha > 0]
//       ... ntrans times
//
// Four things about the reader are easy to get wrong and are reproduced deliberately:
//
// 1. **Field widths truncate.** G4LevelReader::ReadDataItem reads into `char[20]`, `char[14]`
//    and `char[8]` for double, float and int, and `istream >> char(&)[N]` stops after N-1
//    characters WITHOUT consuming the rest - so a 14-character float field is split in two and
//    every subsequent field on the line shifts. read_token() below reproduces the limits so
//    that if the dataset ever contains such a field this port mis-parses it the same way,
//    rather than parsing it correctly and disagreeing.
// 2. **A level can be null.** `vLevel[i]` is left null when the level has no transitions and a
//    non-negative lifetime, so LifeTime(i) is 0 for it while a stable level (lifetime < 0)
//    gets a real object with zero transitions. The difference decides whether photon
//    evaporation calls the fragment long-lived.
// 3. **Broken data is repaired, not rejected.** A level energy below its predecessor is raised
//    to it, and a transition to a level at or above its own index is redirected to the ground
//    state. Both print a warning in Geant4 and both occur in the installed dataset - z89.a219
//    has a transition from level 24 to level 24.
// 4. **The internal-conversion coefficients are read and thrown away.** With
//    fStoreAllLevels = false (11.1.1's default) G4LevelReader parses the ten ICC values only to
//    advance the stream. They still have to be consumed or every following field is wrong.
//
// The spin field packs three things: `spin = 100 + round(2J) + 100000*floating_index`, so
// SpinTwo is |spin%100000 - 100|, the parity is the sign of that difference, and the floating
// index is spin/100000. A negative spin in the file means negative parity.
#ifndef G4GPU_DATA_LEVEL_DATA_CUH
#define G4GPU_DATA_LEVEL_DATA_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/units.cuh"
#include "data/level_index.hh"

namespace g4gpu::data {

namespace u = g4gpu::units;

/// One gamma transition out of a level. Mirrors G4NucLevel's four parallel vectors.
struct LevelTransition {
  int trans;         ///< final level index * 10000 + multipolarity, as G4NucLevel packs it
  float cum_prob;    ///< G4NucLevel::GammaCumProbability - normalised, last entry exactly 1
  float prob;        ///< G4NucLevel::GammaProbability = 1/(1 + alpha), the gamma-not-IC share
  float ratio;       ///< G4NucLevel::MultipolarityRatio
};

/// One nuclear level.
struct NuclearLevel {
  double energy;     ///< MeV
  double time_gamma; ///< ns; negative means stable, and 0 means "no G4NucLevel object"
  int spin;          ///< the packed 100 + 2J + 100000*floating
  int ntrans;        ///< 0 both for a stable level and for one with no G4NucLevel
  int trans_offset;  ///< into the transition array
  bool has_level;    ///< whether G4LevelReader created a G4NucLevel here at all
};

/// One nuclide's G4LevelManager.
struct LevelManagerEntry {
  int z, a;
  int n_levels;          ///< G4LevelManager::NumberOfTransitions() + 1
  int level_offset;      ///< into the level array
  double shell_correction;   ///< G4LevelManager::ShellCorrection(), MeV
  double level_density;      ///< G4LevelManager::LevelDensity(), 1/MeV
};

/// The flat table. Raw pointers so the same struct can be filled on the host and read on the
/// device; `z_a_index` is the (Z, A) -> entry map, sized by the AMIN/AMAX window.
struct LevelTable {
  const LevelManagerEntry* managers = nullptr;
  const NuclearLevel* levels = nullptr;
  const LevelTransition* transitions = nullptr;
  const int* z_a_index = nullptr;   ///< -1 where no manager exists
  int n_managers = 0;
  int n_levels = 0;
  int n_transitions = 0;
};

// ---------------------------------------------------------------------------------------------
// Device-callable accessors: G4NuclearLevelData, G4LevelManager and G4NucLevel.
// ---------------------------------------------------------------------------------------------

/// G4NuclearLevelData::GetMinA / GetMaxA - the compiled window, from level_index.hh.
__host__ __device__ inline int level_min_a(int Z) {
  return (Z >= 0 && Z < kLevelZMax) ? level_amin()[Z] : 0;
}
__host__ __device__ inline int level_max_a(int Z) {
  return (Z >= 0 && Z < kLevelZMax) ? level_amax()[Z] : 0;
}

/// The flat index into z_a_index. Uses the same LEVELIDX slicing the compiled level_max table
/// does, so the two are guaranteed to address the same nuclide.
__host__ __device__ inline int level_flat_index(int Z, int A) {
  if (Z < 1 || Z >= kLevelZMax) { return -1; }
  const int amin = level_amin()[Z], amax = level_amax()[Z];
  if (amax <= 0 || A < amin || A > amax) { return -1; }
  return level_idx()[Z] + A - amin;
}

/// G4NuclearLevelData::GetMaxLevelEnergy - the table COMPILED INTO Geant4, from
/// PhotonEvaporation 5.2, not from the installed dataset. See level_index.hh: for 952 of the
/// 3108 nuclides that have level data these differ, and Geant4 uses this one to gate
/// GetLevelEnergy and to decide which fragments enter the Fermi break-up pool.
__host__ __device__ inline double compiled_max_level_energy(int Z, int A) {
  const int i = level_flat_index(Z, A);
  return (i >= 0) ? static_cast<double>(level_max_energy_MeV()[i]) : 0.0;
}

/// G4NuclearLevelData::MaxLevelEnergy - the float-typed twin of the above, which is what
/// G4FermiFragmentsPoolVI compares against 0.0f.
__host__ __device__ inline float compiled_max_level_energy_f(int Z, int A) {
  const int i = level_flat_index(Z, A);
  return (i >= 0) ? level_max_energy_MeV()[i] : 0.0f;
}

/// The manager for (Z, A), or -1. Unlike Geant4 this cannot read a file on demand: the table
/// is built once, by upload_level_data(), for the whole AMIN/AMAX window - which is what
/// G4ExcitationHandler::SetParameters does anyway through UploadNuclearLevelData(Zmax+1).
__host__ __device__ inline int find_manager(const LevelTable& t, int Z, int A) {
  const int i = level_flat_index(Z, A);
  if (i < 0 || t.z_a_index == nullptr) { return -1; }
  return t.z_a_index[i];
}

/// G4LevelManager::MaxLevelEnergy - the highest level the installed data actually has.
__host__ __device__ inline double read_max_level_energy(const LevelTable& t, int m) {
  const LevelManagerEntry& e = t.managers[m];
  return t.levels[e.level_offset + e.n_levels - 1].energy;
}

/// G4LevelManager::NearestLowEdgeLevelIndex - std::lower_bound on the energies, minus one.
///
/// The `- 1` is not a fencepost slip: lower_bound returns the first level at or above `energy`
/// and the function wants the level at or below it. For an energy equal to a level's the
/// result is therefore that level's index minus one, and NearestLevelIndex's midpoint test
/// puts it back. Reproduced as written, including that an energy below level 0 gives -1 cast
/// to size_t in Geant4 - guarded to 0 here, which is the only place this differs and is the
/// difference between an index and an out-of-bounds read.
__host__ __device__ inline int nearest_low_edge_level_index(const LevelTable& t, int m,
                                                            double energy) {
  const LevelManagerEntry& e = t.managers[m];
  const int ntrans = e.n_levels - 1;
  const NuclearLevel* lv = t.levels + e.level_offset;
  if (energy >= lv[ntrans].energy) { return ntrans; }
  int lo = 0, hi = e.n_levels;   // lower_bound over [0, n_levels)
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (lv[mid].energy < energy) { lo = mid + 1; } else { hi = mid; }
  }
  const int idx = lo - 1;
  return (idx < 0) ? 0 : idx;
}

/// G4LevelManager::NearestLevelIndex(energy, index).
__host__ __device__ inline int nearest_level_index(const LevelTable& t, int m, double energy,
                                                   int index = 0) {
  const LevelManagerEntry& e = t.managers[m];
  const int ntrans = e.n_levels - 1;
  const NuclearLevel* lv = t.levels + e.level_offset;
  int idx = (index < ntrans) ? index : ntrans;
  const double tolerance = 10.0 * u::eV<double>();
  if (ntrans == 0 || fabs(energy - lv[idx].energy) <= tolerance) { return idx; }
  idx = nearest_low_edge_level_index(t, m, energy);
  if (idx < ntrans && (lv[idx].energy + lv[idx + 1].energy) * 0.5 <= energy) { ++idx; }
  return idx;
}

__host__ __device__ inline double level_energy(const LevelTable& t, int m, int i) {
  return t.levels[t.managers[m].level_offset + i].energy;
}

/// G4LevelManager::LifeTime - 0 when the level has no G4NucLevel object.
__host__ __device__ inline double level_lifetime(const LevelTable& t, int m, int i) {
  const NuclearLevel& l = t.levels[t.managers[m].level_offset + i];
  return l.has_level ? l.time_gamma : 0.0;
}

__host__ __device__ inline int level_spin_two(const LevelTable& t, int m, int i) {
  const int s = t.levels[t.managers[m].level_offset + i].spin;
  const int v = s % 100000 - 100;
  return (v < 0) ? -v : v;
}
__host__ __device__ inline int level_parity(const LevelTable& t, int m, int i) {
  const int s = t.levels[t.managers[m].level_offset + i].spin;
  return (s % 100000 - 100 > 0) ? 1 : -1;
}
__host__ __device__ inline int level_floating(const LevelTable& t, int m, int i) {
  return t.levels[t.managers[m].level_offset + i].spin / 100000;
}
__host__ __device__ inline int level_ntrans(const LevelTable& t, int m, int i) {
  const NuclearLevel& l = t.levels[t.managers[m].level_offset + i];
  return l.has_level ? l.ntrans : 0;
}

/// G4NucLevel::SampleGammaTransition - the first index whose cumulative probability is at or
/// above the random number. Falls off the end to `length` in Geant4 if the last cumulative
/// probability is below rndm, which cannot happen because the reader forces it to 1.
__host__ __device__ inline int sample_gamma_transition(const LevelTable& t, int m, int i,
                                                       double rndm) {
  const NuclearLevel& l = t.levels[t.managers[m].level_offset + i];
  const LevelTransition* tr = t.transitions + l.trans_offset;
  const float x = static_cast<float>(rndm);
  int idx = 0;
  for (; idx < l.ntrans; ++idx) {
    if (x <= tr[idx].cum_prob) { break; }
  }
  return idx;
}

__host__ __device__ inline const LevelTransition& level_transition(const LevelTable& t, int m,
                                                                   int i, int j) {
  const NuclearLevel& l = t.levels[t.managers[m].level_offset + i];
  return t.transitions[l.trans_offset + j];
}

/// G4NucLevel::FinalExcitationIndex and ::TransitionType, unpacked from `trans`.
__host__ __device__ inline int transition_final_index(const LevelTransition& tr) {
  return tr.trans / 10000;
}
__host__ __device__ inline int transition_type(const LevelTransition& tr) {
  return tr.trans % 10000;
}

// ---------------------------------------------------------------------------------------------
// Host reader.
// ---------------------------------------------------------------------------------------------

/// Owns the vectors the flat LevelTable points into.
struct LevelTableStorage {
  std::vector<LevelManagerEntry> managers;
  std::vector<NuclearLevel> levels;
  std::vector<LevelTransition> transitions;
  std::vector<int> z_a_index;
  /// Nuclides whose file was present but which produced no manager, and the reason - reported
  /// rather than silently skipped.
  std::vector<std::string> notes;

  LevelTable view() const {
    LevelTable t;
    t.managers = managers.data();
    t.levels = levels.data();
    t.transitions = transitions.data();
    t.z_a_index = z_a_index.data();
    t.n_managers = static_cast<int>(managers.size());
    t.n_levels = static_cast<int>(levels.size());
    t.n_transitions = static_cast<int>(transitions.size());
    return t;
  }
};

namespace detail {

/// One whitespace-delimited token, truncated to `max_chars - 1` characters with the remainder
/// LEFT IN THE STREAM - exactly what `istream >> char(&)[N]` does, and the reason
/// G4LevelReader's 20/14/8-byte buffers are part of the format rather than an implementation
/// detail. Returns false at end of file.
inline bool read_token(std::FILE* f, char* out, int max_chars) {
  int c;
  do { c = std::fgetc(f); } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
  if (c == EOF) { out[0] = '\0'; return false; }
  int n = 0;
  while (c != EOF && c != ' ' && c != '\t' && c != '\n' && c != '\r' && n < max_chars - 1) {
    out[n++] = static_cast<char>(c);
    c = std::fgetc(f);
  }
  out[n] = '\0';
  // Push back the delimiter (or the first character past the buffer limit) so the next token
  // starts where Geant4's would.
  if (c != EOF) { std::ungetc(c, f); }
  return true;
}

inline bool read_double(std::FILE* f, double& x) {
  char buf[20];
  x = 0.0;
  if (!read_token(f, buf, 20)) { return false; }
  x = std::strtod(buf, nullptr);
  return true;
}
inline bool read_float(std::FILE* f, float& x) {
  char buf[14];
  x = 0.0f;
  if (!read_token(f, buf, 14)) { return false; }
  x = static_cast<float>(std::atof(buf));
  return true;
}
inline bool read_int(std::FILE* f, int& x) {
  char buf[8];
  x = 0;
  if (!read_token(f, buf, 8)) { return false; }
  x = std::atoi(buf);
  return true;
}
/// G4LevelReader reads the floating-level marker with `infile >> fPol` on a G4String, so the
/// whole token arrives - unlike ReadDataItem(G4String&), which is limited to two characters
/// and is not the call used here.
inline bool read_pol(std::FILE* f, char* out, int cap) {
  return read_token(f, out, cap);
}

/// G4LevelReader::fFloatingLevels, in order. The index is what goes into the spin word.
inline int floating_index(const char* s) {
  static const char* names[13] = {"-", "+X", "+Y", "+Z", "+U", "+V", "+W",
                                  "+R", "+S", "+T", "+A", "+B", "+C"};
  for (int k = 0; k < 13; ++k) {
    if (std::string(names[k]) == s) { return k; }
  }
  // Geant4's loop leaves k == 13 when nothing matches, and 13 is then multiplied into the spin
  // word. Reproduced: an unknown marker is not an error, it is a floating index of 13.
  return 13;
}

}  // namespace detail

/// G4LevelReader::LevelManager for one nuclide. Appends to `st` and returns the manager index,
/// or -1 when the file is absent or yields no level.
///
/// `shell_corr` and `lev_density` are the two numbers the G4LevelManager constructor computes
/// and caches; they are passed in rather than computed here so that src/data/ does not depend
/// on the physics headers.
inline int read_level_manager(LevelTableStorage& st, const std::string& dir, int Z, int A,
                              double shell_corr, double lev_density) {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/z%d.a%d", dir.c_str(), Z, A);
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) {
    // G4LevelReader raises a FatalException for Z < 6 and returns null otherwise. The
    // exception is not reproduced as a crash, but the absence is recorded so that a missing
    // low-Z file is visible rather than silently becoming "this nuclide has no levels".
    if (Z < 6) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "MISSING FILE for Z=%d A=%d (Geant4 aborts here)", Z, A);
      st.notes.push_back(msg);
    }
    return -1;
  }

  const int level_offset = static_cast<int>(st.levels.size());
  const int trans_offset0 = static_cast<int>(st.transitions.size());
  // G4LevelReader::fTimeFactor = second / logZ(2). Its unit is a lifetime multiplier: the
  // dataset stores half-lives in seconds and the code wants mean lives in G4 time units.
  const double time_factor = u::s<double>() / std::log(2.0);
  const float alpha_max = 1.0e15f;

  int nread = 0;
  double prev_energy = 0.0;
  bool broken = false;
  std::vector<LevelTransition> trans_buf;

  // G4LevelReader::fLevelMax, 632. It reads at most that many levels per nuclide and, on this
  // path (nlev == 0), never grows: `nlevels = fLevelMax` and the resize is guarded by
  // `nlevels > fLevelMax`. Not a slack buffer size either - z18.a38 has exactly 632 levels and
  // is the largest file in PhotonEvaporation5.7, so the constant is the dataset's maximum and
  // a 633-level nuclide would be silently truncated. Reproduced for that reason.
  const int kLevelMax = 632;

  for (int i = 0; i < kLevelMax; ++i) {
    char idx_tok[8], pol[16];
    if (!detail::read_token(f, idx_tok, 8)) { break; }
    const int i1 = std::atoi(idx_tok);
    if (!detail::read_pol(f, pol, 16)) { break; }
    if (i1 != i) {
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "Z=%d A=%d level #%d has index %d - stopping, as G4LevelReader does",
                    Z, A, i, i1);
      st.notes.push_back(msg);
      break;
    }
    double ener = 0.0, ftime = 0.0;
    float fspin = 0.0f;
    int ntrans = 0;
    if (!(detail::read_double(f, ener) && detail::read_double(f, ftime) &&
          detail::read_float(f, fspin) && detail::read_int(f, ntrans))) {
      break;
    }
    ener *= u::keV<double>();
    const int k = detail::floating_index(pol);
    if (i > 0 && ener < prev_energy) {
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "Z=%d A=%d broken level %d E=%g < %g - energy raised, as Geant4 does",
                    Z, A, i, ener, prev_energy);
      st.notes.push_back(msg);
      ener = prev_energy;
    }
    prev_energy = ener;
    if (ftime > 0.0) { ftime *= time_factor; }
    if (fspin > 48.0f) { fspin = 0.0f; }
    // G4lrint - round half to even, which is what std::lrint does under the default rounding
    // mode. The values in the dataset are half-integers, so the tie case does occur.
    const int twos = static_cast<int>(std::lrint(2.0 * static_cast<double>(fspin)));

    NuclearLevel lev;
    lev.energy = ener;
    lev.time_gamma = ftime;
    lev.spin = 100 + twos + k * 100000;
    lev.ntrans = 0;
    lev.trans_offset = static_cast<int>(st.transitions.size());
    lev.has_level = false;

    if (ntrans == 0 && ftime < 0.0) {
      // A stable level with no transitions still gets a G4NucLevel, with zero transitions -
      // which is how LifeTime(i) comes back negative for it instead of zero.
      lev.has_level = true;
    } else if (ntrans > 0) {
      trans_buf.clear();
      float norm1 = 0.0f;
      int jdone = 0;
      for (int j = 0; j < ntrans; ++j) {
        int i2 = 0, tnum = 0;
        double tener = 0.0;
        float fprob = 0.0f, ratio = 0.0f, falpha = 0.0f;
        if (!(detail::read_int(f, i2) && detail::read_double(f, tener) &&
              detail::read_float(f, fprob) && detail::read_int(f, tnum) &&
              detail::read_float(f, ratio) && detail::read_float(f, falpha))) {
          // REFUSED, not reproduced. Geant4 breaks out of this loop and then builds a
          // G4NucLevel with the DECLARED ntrans anyway, so the transitions past the failure
          // hold whatever the reader's reusable vectors held for the previous nuclide - a
          // value that depends on the order the files were read in. That is not something a
          // port can reproduce, and it is not something a physics answer should depend on.
          // It requires a truncated file: no nuclide in PhotonEvaporation5.7 reaches it, and
          // read_all_level_data() fails loudly if one ever does.
          char msg[176];
          std::snprintf(msg, sizeof(msg),
                        "REFUSED Z=%d A=%d level %d: transition %d of %d is truncated; "
                        "Geant4 would use uninitialised data here", Z, A, i, j, ntrans);
          st.notes.push_back(msg);
          broken = true;
          break;
        }
        if (i2 >= i) {
          char msg[176];
          std::snprintf(msg, sizeof(msg),
                        "Z=%d A=%d broken transition %d from level %d to %d - ground level used",
                        Z, A, j, i, i2);
          st.notes.push_back(msg);
          i2 = 0;
        }
        if (falpha < 0.0f) { falpha = 0.0f; }
        if (falpha > alpha_max) { falpha = alpha_max; }
        const float x = 1.0f + falpha;
        norm1 += x * fprob;
        LevelTransition tr;
        tr.trans = i2 * 10000 + tnum;
        tr.cum_prob = norm1;
        tr.prob = 1.0f / x;
        tr.ratio = ratio;
        trans_buf.push_back(tr);
        ++jdone;
        if (falpha > 0.0f) {
          // The ten internal-conversion coefficients. Read and discarded: with
          // fStoreAllLevels false, G4LevelReader does not build a shell-probability table.
          // They must still be consumed or the next line's fields are read from the middle of
          // this one. Geant4 stops early if one fails to read and zeroes the rest.
          for (int c = 0; c < 10; ++c) {
            float icc = 0.0f;
            if (!detail::read_float(f, icc)) { break; }
          }
        }
      }
      if (broken) { break; }
      if (jdone > 0) {
        if (norm1 > 0.0f) { norm1 = 1.0f / norm1; }
        // Geant4 normalises indices 0..ntrans-2 and forces index ntrans-1 to exactly 1. The
        // forcing is why SampleGammaTransition can never fall off the end, and why a level
        // whose intensities are all zero still samples its last transition.
        const int nt = jdone - 1;
        for (int c = 0; c < nt; ++c) { trans_buf[c].cum_prob *= norm1; }
        trans_buf[nt].cum_prob = 1.0f;
        for (const LevelTransition& tr : trans_buf) { st.transitions.push_back(tr); }
        lev.ntrans = jdone;
        lev.has_level = true;
      }
    }
    st.levels.push_back(lev);
    nread = i + 1;
  }
  std::fclose(f);

  if (nread < 1 || broken) {
    st.levels.resize(level_offset);
    st.transitions.resize(trans_offset0);
    return -1;
  }
  LevelManagerEntry e;
  e.z = Z;
  e.a = A;
  e.n_levels = nread;
  e.level_offset = level_offset;
  e.shell_correction = shell_corr;
  e.level_density = lev_density;
  st.managers.push_back(e);
  return static_cast<int>(st.managers.size()) - 1;
}

/// G4NuclearLevelData::UploadNuclearLevelData - reads every nuclide in the AMIN/AMAX window
/// for Z = 1 .. zmax-1, which is what G4ExcitationHandler::SetParameters asks for with
/// `Zmax + 1` where Zmax is the largest Z in the geometry (floored at 20).
///
/// Note the strict `Z < mZ`: the element with the largest Z in the geometry is loaded only
/// because SetParameters passes Zmax + 1. Reproduced, since the pool of fragments Fermi
/// break-up builds depends on which managers exist.
///
/// `shell_corr_fn` and `ld_fn` supply the two cached numbers of the G4LevelManager
/// constructor. They are callbacks so this file stays free of the physics headers, which
/// depend on it.
template <typename ShellFn, typename LdFn>
inline void read_all_level_data(LevelTableStorage& st, const std::string& dir, int zmax,
                                ShellFn shell_corr_fn, LdFn ld_fn) {
  st.z_a_index.assign(kLevelMaxEntries, -1);
  const int mz = (zmax > kLevelZMax) ? kLevelZMax : zmax;
  for (int Z = 1; Z < mz; ++Z) {
    const int amin = level_min_a(Z), amax = level_max_a(Z);
    if (amax <= 0) { continue; }
    for (int A = amin; A <= amax; ++A) {
      const int m = read_level_manager(st, dir, Z, A, shell_corr_fn(Z, A), ld_fn(Z, A));
      if (m >= 0) {
        const int fi = level_flat_index(Z, A);
        if (fi >= 0) { st.z_a_index[fi] = m; }
      }
    }
  }
}

}  // namespace g4gpu::data

#endif
