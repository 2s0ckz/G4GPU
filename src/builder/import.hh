// Importers for CAD meshes and voxel volumes.
//
// What is supported, and what is not, stated plainly because a half-working importer that
// silently produces an empty or wrong volume is worse than one that refuses:
//
//   .stl        binary and ASCII triangle meshes. Fully read.
//   .raw .bin   raw voxel arrays. The dimensions and the element type cannot be inferred from
//               the file - there is no header - so they are either given alongside the path or
//               guessed from the file size and reported for confirmation.
//   .mat        MATLAB level 5, uncompressed, a single numeric array. Level 7.3 files are
//               HDF5 and are rejected with that explanation.
//   .h5         not supported: HDF5 is a library dependency this project does not take. The
//               error says so and says what to convert to.
//
// A discrete volume's distinct values become material classes; a continuous one is banded by
// Hounsfield number. Which of the two a file is cannot be determined from the data - a CT with
// few distinct values looks discrete - so the caller asks.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "geometry/mesh_io.hh"

#include "builder/model.hh"

namespace g4gpu::builder {

/// A voxel array as read from a file.
struct VoxelData {
  int nx = 0, ny = 0, nz = 0;
  std::vector<float> value;  ///< nx*ny*nz, x fastest
  std::string note;          ///< how the shape and type were determined, for the log
};

enum class VoxelType : int { kUInt8, kInt16, kUInt16, kInt32, kFloat32, kFloat64 };

inline int VoxelTypeSize(VoxelType t) {
  switch (t) {
    case VoxelType::kUInt8: return 1;
    case VoxelType::kInt16:
    case VoxelType::kUInt16: return 2;
    case VoxelType::kInt32:
    case VoxelType::kFloat32: return 4;
    case VoxelType::kFloat64: return 8;
  }
  return 1;
}

/// Reads a raw voxel array. If `nx` is zero the dimensions are guessed from the file size by
/// assuming a cube, which is right often enough to be worth offering and always reported.
bool ReadRawVoxels(const std::string& path, VoxelType type, int nx, int ny, int nz,
                   VoxelData& out);

/// Reads a single numeric array from an uncompressed MATLAB level-5 .mat file.
bool ReadMatVoxels(const std::string& path, VoxelData& out);

/// Reads an STL mesh into a flat triangle list (9 floats per triangle).
bool ReadMesh(const std::string& path, std::vector<float>& triangles, std::string& note);
bool ReadStl(const std::string& path, std::vector<float>& triangles, std::string& note);

// ---------------------------------------------------------------- implementation

namespace detail {

inline bool EndsWith(const std::string& s, const char* suffix) {
  const std::size_t n = std::strlen(suffix);
  if (s.size() < n) { return false; }
  for (std::size_t i = 0; i < n; ++i) {
    if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i]))
        != std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

/// The last component of a path, for showing a chosen file in a dialog where the full path
/// would not fit and its directory is not what the user is checking.
inline std::string BaseName(const std::string& p) {
  const std::size_t a = p.find_last_of('/');
  const std::size_t b = p.find_last_of('\\');
  std::size_t at = std::string::npos;
  if (a != std::string::npos) { at = a; }
  if (b != std::string::npos && (at == std::string::npos || b > at)) { at = b; }
  return (at == std::string::npos) ? p : p.substr(at + 1);
}

}  // namespace detail

inline bool ReadRawVoxels(const std::string& path, VoxelType type, int nx, int ny, int nz,
                          VoxelData& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::printf("cannot open %s\n", path.c_str());
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);

  const int esz = VoxelTypeSize(type);
  char shape_note[192] = {0};
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    // No header, so the shape has to come from somewhere. A cube is the common case for a
    // phantom dump; anything else has to be given explicitly, and the note says which was
    // assumed so a wrong guess is visible immediately rather than as a scrambled volume.
    const long long n = bytes / esz;
    const long long side = static_cast<long long>(std::llround(std::cbrt(
        static_cast<double>(n))));
    if (side > 0 && side * side * side == n) {
      nx = ny = nz = static_cast<int>(side);
      std::snprintf(shape_note, sizeof shape_note,
                    "no header; %lld elements is exactly %lldx%lldx%lld, assumed cubic", n,
                    side, side, side);
    } else {
      std::fclose(f);
      std::printf("%s: %lld bytes / %d = %lld elements, which is not a cube.\n"
                  "  Give the dimensions explicitly - a raw file carries no header.\n",
                  path.c_str(), bytes, esz, n);
      return false;
    }
  } else {
    const long long want = static_cast<long long>(nx) * ny * nz * esz;
    if (want != bytes) {
      std::fclose(f);
      std::printf("%s: %dx%dx%d of %d-byte elements is %lld bytes, but the file is %lld.\n",
                  path.c_str(), nx, ny, nz, esz, want, bytes);
      return false;
    }
    std::snprintf(shape_note, sizeof shape_note, "%dx%dx%d as given", nx, ny, nz);
  }

  const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
  std::vector<unsigned char> raw(n * esz);
  const std::size_t got = std::fread(raw.data(), 1, raw.size(), f);
  std::fclose(f);
  if (got != raw.size()) {
    std::printf("%s: short read (%zu of %zu bytes)\n", path.c_str(), got, raw.size());
    return false;
  }

  out.nx = nx;
  out.ny = ny;
  out.nz = nz;
  out.value.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned char* p = raw.data() + i * esz;
    switch (type) {
      case VoxelType::kUInt8: out.value[i] = static_cast<float>(*p); break;
      case VoxelType::kInt16: {
        std::int16_t v;
        std::memcpy(&v, p, 2);
        out.value[i] = static_cast<float>(v);
        break;
      }
      case VoxelType::kUInt16: {
        std::uint16_t v;
        std::memcpy(&v, p, 2);
        out.value[i] = static_cast<float>(v);
        break;
      }
      case VoxelType::kInt32: {
        std::int32_t v;
        std::memcpy(&v, p, 4);
        out.value[i] = static_cast<float>(v);
        break;
      }
      case VoxelType::kFloat32: {
        float v;
        std::memcpy(&v, p, 4);
        out.value[i] = v;
        break;
      }
      case VoxelType::kFloat64: {
        double v;
        std::memcpy(&v, p, 8);
        out.value[i] = static_cast<float>(v);
        break;
      }
    }
  }
  out.note = shape_note;
  return true;
}

inline bool ReadMatVoxels(const std::string& path, VoxelData& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::printf("cannot open %s\n", path.c_str());
    return false;
  }
  char header[128] = {0};
  if (std::fread(header, 1, 128, f) != 128) {
    std::fclose(f);
    std::printf("%s: too short to be a .mat file\n", path.c_str());
    return false;
  }
  // A v7.3 file is an HDF5 file with a MATLAB banner; the banner says so.
  if (std::strstr(header, "MATLAB 7.3") != nullptr) {
    std::fclose(f);
    std::printf("%s is a MATLAB v7.3 file, which is HDF5 underneath.\n"
                "  This build does not link HDF5. Re-save it as -v7 or -v6 in MATLAB\n"
                "  (save('f.mat','a','-v6')), or export a .raw.\n",
                path.c_str());
    return false;
  }
  if (std::strncmp(header, "MATLAB", 6) != 0) {
    std::fclose(f);
    std::printf("%s: not a MATLAB level-5 file\n", path.c_str());
    return false;
  }

  // Walk the top-level data elements looking for a numeric miMATRIX.
  while (true) {
    std::uint32_t tag[2];
    if (std::fread(tag, 4, 2, f) != 2) { break; }
    const std::uint32_t type = tag[0] & 0xFFFF;
    const std::uint32_t bytes = tag[1];
    const long long next = std::ftell(f) + ((bytes + 7) / 8) * 8;

    if (type == 15) {  // miCOMPRESSED
      std::fclose(f);
      std::printf("%s holds compressed data.\n"
                  "  Save it with save('f.mat','a','-v6') to write it uncompressed, or\n"
                  "  export a .raw file.\n",
                  path.c_str());
      return false;
    }
    if (type != 14) {  // not miMATRIX
      std::fseek(f, next, SEEK_SET);
      continue;
    }

    // Array flags.
    std::uint32_t sub[2];
    if (std::fread(sub, 4, 2, f) != 2) { break; }
    std::uint32_t flags[2];
    if (std::fread(flags, 4, 2, f) != 2) { break; }
    const std::uint32_t klass = flags[0] & 0xFF;

    // Dimensions.
    if (std::fread(sub, 4, 2, f) != 2) { break; }
    const int ndim = static_cast<int>(sub[1] / 4);
    std::vector<std::int32_t> dims(static_cast<std::size_t>(ndim));
    if (std::fread(dims.data(), 4, static_cast<std::size_t>(ndim), f)
        != static_cast<std::size_t>(ndim)) {
      break;
    }
    if ((ndim * 4) % 8 != 0) { std::fseek(f, 4, SEEK_CUR); }  // pad to 8

    // Name.
    if (std::fread(sub, 4, 2, f) != 2) { break; }
    const std::uint32_t name_bytes = ((sub[0] >> 16) != 0) ? (sub[0] >> 16) : sub[1];
    if ((sub[0] >> 16) == 0) { std::fseek(f, ((name_bytes + 7) / 8) * 8, SEEK_CUR); }

    // Real part.
    if (std::fread(sub, 4, 2, f) != 2) { break; }
    const std::uint32_t dtype = sub[0] & 0xFFFF;
    const std::uint32_t dbytes = sub[1];

    if (klass == 6 || (klass >= 8 && klass <= 13)) {  // double or an integer class
      std::size_t n = 1;
      for (std::int32_t d : dims) { n *= static_cast<std::size_t>(d); }
      std::vector<unsigned char> raw(dbytes);
      if (std::fread(raw.data(), 1, dbytes, f) != dbytes) { break; }
      out.value.resize(n);
      // The miTYPE codes that matter: 1 int8, 2 uint8, 3 int16, 4 uint16, 5 int32, 6 uint32,
      // 7 single, 9 double.
      for (std::size_t i = 0; i < n; ++i) {
        const unsigned char* p = raw.data();
        switch (dtype) {
          case 1: out.value[i] = static_cast<float>(static_cast<std::int8_t>(p[i])); break;
          case 2: out.value[i] = static_cast<float>(p[i]); break;
          case 3: {
            std::int16_t v;
            std::memcpy(&v, p + 2 * i, 2);
            out.value[i] = v;
            break;
          }
          case 4: {
            std::uint16_t v;
            std::memcpy(&v, p + 2 * i, 2);
            out.value[i] = v;
            break;
          }
          case 5: {
            std::int32_t v;
            std::memcpy(&v, p + 4 * i, 4);
            out.value[i] = static_cast<float>(v);
            break;
          }
          case 7: {
            float v;
            std::memcpy(&v, p + 4 * i, 4);
            out.value[i] = v;
            break;
          }
          case 9: {
            double v;
            std::memcpy(&v, p + 8 * i, 8);
            out.value[i] = static_cast<float>(v);
            break;
          }
          default:
            std::fclose(f);
            std::printf("%s: unsupported .mat element type %u\n", path.c_str(), dtype);
            return false;
        }
      }
      out.nx = (ndim > 0) ? dims[0] : 1;
      out.ny = (ndim > 1) ? dims[1] : 1;
      out.nz = (ndim > 2) ? dims[2] : 1;
      char note[160];
      std::snprintf(note, sizeof note, "MATLAB level 5, %dx%dx%d, element type %u", out.nx,
                    out.ny, out.nz, dtype);
      out.note = note;
      std::fclose(f);
      return true;
    }
    std::fseek(f, next, SEEK_SET);
  }
  std::fclose(f);
  std::printf("%s: no uncompressed numeric array found\n", path.c_str());
  return false;
}

/// Reads an STL, Wavefront OBJ or ASCII PLY into flat float triples.
///
/// The reading itself is in geometry/mesh_io.hh, which is also what CADMesh.hh and
/// G4TessellatedSolid use: one implementation, so a format the GUI can import is a format a
/// detector description can import and vice versa. Floats here because the model keeps
/// meshes as floats - a CAD import is a shape, not a measurement, and 24 bits of mantissa is
/// 6 significant figures on a metre-scale part.
inline bool ReadMesh(const std::string& path, std::vector<float>& tri, std::string& note) {
  g4gpu::geom::MeshTriangles d;
  if (!g4gpu::geom::ReadMeshFile(path, d, note)) { return false; }
  tri.clear();
  tri.reserve(d.size());
  for (double v : d) { tri.push_back(static_cast<float>(v)); }
  return !tri.empty();
}

/// Kept for callers that know they have an STL.
inline bool ReadStl(const std::string& path, std::vector<float>& tri, std::string& note) {
  return ReadMesh(path, tri, note);
}

// ---------------------------------------------------------------- colormaps

/// One stop of a colormap: a position in 0-1 and the colour there.
struct ColourStop {
  double t;
  unsigned char r, g, b;
};

/// The colormaps offered at import, in the order the dialog lists them.
///
/// Held as a handful of stops with linear interpolation between them rather than as 256-entry
/// tables: nine stops reproduce the shape of these maps to within a couple of levels, which
/// is finer than the eye resolves in a phantom, and a 256-entry table per map is eight
/// kilobytes of hand-typed numbers with no way to check any one of them.
///
/// The first entry is the hue sweep this file has always used, kept and kept FIRST so that an
/// import which does not touch the menu comes out exactly as it did before. It is the best of
/// them for telling one organ from its neighbour - adjacent classes land far apart on the
/// colour circle - which is not what the perceptual maps are built for. Those represent a
/// CONTINUUM, and they are here because a phantom's indices are often ordered (outwards from
/// the skin, or by tissue density) and reading it as a continuum is then the clearer picture.
enum class Colormap : int {
  kHueSweep = 0, kViridis, kPlasma, kInferno, kMagma, kTurbo, kJet, kGray, kCount
};

inline const char* const* ColormapNames() {
  static const char* const kNames[] = {"distinct hues", "viridis", "plasma", "inferno",
                                       "magma",         "turbo",   "jet",    "gray"};
  return kNames;
}

/// The stops of @p m, or null for the hue sweep, which is computed rather than tabulated.
inline const ColourStop* ColormapStops(Colormap m, int& n) {
  static const ColourStop kViridis[] = {
      {0.000, 68, 1, 84},    {0.125, 71, 44, 122},  {0.250, 59, 81, 139},
      {0.375, 44, 113, 142}, {0.500, 33, 144, 140}, {0.625, 39, 173, 129},
      {0.750, 92, 200, 99},  {0.875, 170, 220, 50}, {1.000, 253, 231, 37}};
  static const ColourStop kPlasma[] = {
      {0.000, 13, 8, 135},   {0.125, 75, 3, 161},   {0.250, 125, 3, 168},
      {0.375, 168, 34, 150}, {0.500, 203, 70, 121}, {0.625, 229, 107, 93},
      {0.750, 248, 148, 65}, {0.875, 253, 195, 40}, {1.000, 240, 249, 33}};
  static const ColourStop kInferno[] = {
      {0.000, 0, 0, 4},      {0.143, 31, 12, 72},   {0.286, 85, 15, 109},
      {0.429, 136, 34, 106}, {0.571, 186, 54, 85},  {0.714, 227, 89, 51},
      {0.857, 249, 142, 9},  {1.000, 252, 255, 164}};
  static const ColourStop kMagma[] = {
      {0.000, 0, 0, 4},      {0.143, 28, 16, 68},   {0.286, 79, 18, 123},
      {0.429, 129, 37, 129}, {0.571, 181, 54, 122}, {0.714, 229, 80, 100},
      {0.857, 251, 135, 97}, {1.000, 252, 253, 191}};
  static const ColourStop kTurbo[] = {
      {0.000, 48, 18, 59},   {0.125, 65, 69, 171},  {0.250, 57, 131, 228},
      {0.375, 27, 184, 203}, {0.500, 54, 220, 140}, {0.625, 140, 244, 64},
      {0.750, 215, 227, 37}, {0.875, 254, 168, 49}, {1.000, 122, 4, 3}};
  // Jet's stops are NOT evenly spaced, and that is what gives it the flat cyan and yellow
  // plateaus it is known and criticised for, so they are written where they actually are.
  static const ColourStop kJet[] = {
      {0.000, 0, 0, 128},   {0.125, 0, 0, 255}, {0.375, 0, 255, 255},
      {0.625, 255, 255, 0}, {0.875, 255, 0, 0}, {1.000, 128, 0, 0}};
  static const ColourStop kGray[] = {{0.000, 0, 0, 0}, {1.000, 255, 255, 255}};

  switch (m) {
    case Colormap::kViridis: n = 9; return kViridis;
    case Colormap::kPlasma:  n = 9; return kPlasma;
    case Colormap::kInferno: n = 8; return kInferno;
    case Colormap::kMagma:   n = 8; return kMagma;
    case Colormap::kTurbo:   n = 9; return kTurbo;
    case Colormap::kJet:     n = 6; return kJet;
    case Colormap::kGray:    n = 2; return kGray;
    default:                 n = 0; return nullptr;
  }
}

/// The colour of @p m at @p t in 0-1, as three components in 0-1.
inline void SampleColormap(Colormap m, double t, float& r, float& g, float& b) {
  if (t < 0.0) { t = 0.0; }
  if (t > 1.0) { t = 1.0; }
  if (m == Colormap::kHueSweep) {
    // The hue circle, swept ALMOST once. A saturated ramp rather than a full HSV sweep, so
    // that no class comes out near-black or near-white - the two the eye cannot place in a
    // phantom.
    //
    // 330 degrees and not 360, because the circle closes: 360 is the same red as 0, so the
    // lowest and highest index would come out identical. The version of this that indexed by
    // POSITION in the class list avoided it by accident - the last of N classes landed at
    // 360*(N-1)/N, one step short - and indexing by value instead put a class exactly at the
    // end and closed the loop. The test that caught it is section 13 of
    // tests/test_voxel_import.cu, which asks whether any map returns to a colour it has left.
    const double hh = (t * 330.0) / 60.0;
    const int i = static_cast<int>(hh) % 6;
    const double fr = hh - static_cast<int>(hh);
    const double p = 0.35, qv = 1.0 - 0.65 * fr, tt = 0.35 + 0.65 * fr;
    switch (i) {
      case 0: r = 1.0f; g = static_cast<float>(tt); b = static_cast<float>(p); break;
      case 1: r = static_cast<float>(qv); g = 1.0f; b = static_cast<float>(p); break;
      case 2: r = static_cast<float>(p); g = 1.0f; b = static_cast<float>(tt); break;
      case 3: r = static_cast<float>(p); g = static_cast<float>(qv); b = 1.0f; break;
      case 4: r = static_cast<float>(tt); g = static_cast<float>(p); b = 1.0f; break;
      default: r = 1.0f; g = static_cast<float>(p); b = static_cast<float>(qv); break;
    }
    return;
  }
  int n = 0;
  const ColourStop* st = ColormapStops(m, n);
  if (st == nullptr || n <= 0) {
    r = 0.7f;
    g = 0.7f;
    b = 0.7f;
    return;
  }
  if (t <= st[0].t || n == 1) {
    r = st[0].r / 255.0f;
    g = st[0].g / 255.0f;
    b = st[0].b / 255.0f;
    return;
  }
  for (int k = 1; k < n; ++k) {
    if (t <= st[k].t || k == n - 1) {
      const double span = st[k].t - st[k - 1].t;
      const double f = (span > 0) ? (t - st[k - 1].t) / span : 0.0;
      r = static_cast<float>((st[k - 1].r + (st[k].r - st[k - 1].r) * f) / 255.0);
      g = static_cast<float>((st[k - 1].g + (st[k].g - st[k - 1].g) * f) / 255.0);
      b = static_cast<float>((st[k - 1].b + (st[k].b - st[k - 1].b) * f) / 255.0);
      return;
    }
  }
}

/// What reading a colour table did, for the log.
struct ColourTableResult {
  int rows = 0;        ///< data rows parsed
  int applied = 0;     ///< rows whose index named a class
  int unmatched = 0;   ///< rows whose index named nothing
  int malformed = 0;   ///< lines with four-plus fields that are not all numbers
  int on_unit = 0;     ///< values read as 0-1
  int on_255 = 0;      ///< values read as 0-255
  int clamped = 0;
  bool ok = false;
};

/// Reads a per-class colour table: one row per voxel class, `index r g b [a]`.
///
/// The index is the class's VALUE - the number in the segmentation file - and not its position
/// in the list, because that is what a segmentation's own colour table is keyed by. A row whose
/// index names no class is counted and skipped rather than silently dropped: a table from a
/// different phantom is a mistake worth being told about, and it looks exactly like a table
/// that partly matches.
///
/// SCALE IS DECIDED BY THE WAY EACH NUMBER IS WRITTEN. `0.5` is a half; `128` is a half. A
/// token holding a decimal point or an exponent is read on 0-1, anything else on 0-255. That
/// is the convention colour tables in the wild use and it is unambiguous per value - which
/// matters, because the alternative rules are all worse: a per-file guess from the maximum
/// value cannot tell a 0-255 table that happens to top out at 1 from a 0-1 table, and asking
/// the user is a question they should not have to answer about a file that already says.
///
/// The edge is real and is worth knowing: `1` is 1/255, very nearly black, and `1.0` is full.
/// Both counts are reported so a table read the wrong way is visible in the log rather than
/// only in the picture.
inline ColourTableResult ReadVoxelColourTable(const std::string& path, Solid& s,
                                              std::string& note) {
  ColourTableResult r;
  char buf[256];
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    note = "could not open " + path;
    return r;
  }
  std::string text;
  {
    char chunk[4096];
    std::size_t n = 0;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) { text.append(chunk, n); }
  }
  std::fclose(f);

  // A number, or false. strtod rather than atof because atof cannot fail: it returns zero for
  // a word, so the header line `index,r,g,b,a` would read as class 0 painted black. Requiring
  // the whole token to be consumed is what tells a header from a row.
  auto number = [](const std::string& tok, double& out) {
    if (tok.empty()) { return false; }
    char* end = nullptr;
    out = std::strtod(tok.c_str(), &end);
    // `nan` and `inf` parse whole, and a NaN colour is undefined behaviour by the time the
    // scene builder casts it to a byte, so they are not numbers for this purpose.
    return end == tok.c_str() + tok.size() && std::isfinite(out);
  };

  // One channel, on whichever scale it was written in.
  auto scale = [&r](const std::string& tok, double v) {
    const bool fractional = tok.find('.') != std::string::npos
                            || tok.find('e') != std::string::npos
                            || tok.find('E') != std::string::npos;
    double u = fractional ? v : v / 255.0;
    if (fractional) { ++r.on_unit; } else { ++r.on_255; }
    if (u < 0.0) { u = 0.0; ++r.clamped; }
    if (u > 1.0) { u = 1.0; ++r.clamped; }
    return static_cast<float>(u);
  };

  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t nl = text.find('\n', at);
    std::string line = text.substr(at, (nl == std::string::npos) ? std::string::npos : nl - at);
    at = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
    // Trim, then skip blanks and the three comment markers a table might use.
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }
    std::size_t b = 0;
    while (b < line.size() && (line[b] == ' ' || line[b] == '\t')) { ++b; }
    line = line.substr(b);
    if (line.empty() || line[0] == '#' || line[0] == ';'
        || (line.size() > 1 && line[0] == '/' && line[1] == '/')) {
      continue;
    }
    // Commas or whitespace, either or both: a table is as likely to be CSV as columns.
    std::vector<std::string> tok;
    std::string cur;
    for (char ch : line) {
      if (ch == ',' || ch == ' ' || ch == '\t') {
        if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
      } else {
        cur.push_back(ch);
      }
    }
    if (!cur.empty()) { tok.push_back(cur); }
    if (tok.size() < 4) { continue; }   // not a row: too few fields to be one

    // Every field a number, or the line is not a row. Counted, because a table with a stray
    // word in it is a table the user should look at rather than one that half worked.
    double num[5] = {0, 0, 0, 0, 0};
    const std::size_t want = (tok.size() >= 5) ? 5 : 4;
    bool all = true;
    for (std::size_t k = 0; k < want; ++k) {
      if (!number(tok[k], num[k])) { all = false; break; }
    }
    if (!all) {
      ++r.malformed;
      continue;
    }
    ++r.rows;
    const float rr = scale(tok[1], num[1]);
    const float gg = scale(tok[2], num[2]);
    const float bb = scale(tok[3], num[3]);
    const bool have_a = (want == 5);
    const float aa = have_a ? scale(tok[4], num[4]) : 0.0f;

    // Which class the row names.
    //
    // For a segmentation - the case this exists for - it is the VALUE in the file, because
    // that is what the phantom's own colour table is keyed by, and a class list sorted or
    // filtered differently must still take the same colours. For continuous BANDS there is no
    // such external number: a band is `>= 300 HU`, invented here rather than read, so a row
    // there names the band by position, 0 for the first.
    VoxelClass* target = nullptr;
    if (s.voxel_kind == VoxelKind::kDiscrete) {
      for (VoxelClass& c : s.voxel_classes) {
        if (c.value == num[0]) {
          target = &c;
          break;
        }
      }
    } else if (num[0] >= 0 && num[0] < static_cast<double>(s.voxel_classes.size())) {
      target = &s.voxel_classes[static_cast<std::size_t>(num[0])];
    }
    if (target == nullptr) {
      ++r.unmatched;
      continue;
    }
    target->r = rr;
    target->g = gg;
    target->b = bb;
    if (have_a) {
      // THE NUMBER THE TABLE GIVES, AND NOTHING ELSE. This used to also clear `visible` when
      // the alpha was zero, on the reasoning that a table calling a class transparent is
      // saying it should not be drawn and the checkbox beside it would otherwise contradict
      // the number. That is a second decision made on the user's behalf, and it is the same
      // mistake as hiding index 0 on import: `visible` is a control the user owns, and a class
      // whose eye is shut without them shutting it is a class they have to discover before
      // they can wonder where it went.
      //
      // Zero opacity already draws nothing, so nothing is lost by leaving the checkbox alone -
      // and raising the opacity then works on its own, instead of appearing to do nothing
      // because a flag they never touched is still off.
      target->opacity = aa;
    }
    ++r.applied;
  }

  r.ok = (r.applied > 0);
  if (!r.ok) {
    if (r.rows == 0) {
      std::snprintf(buf, sizeof buf,
                    "no rows in this file. A row is `index r g b [a]`, comma or space "
                    "separated;\n  %d lines had four or more fields but were not numbers.",
                    r.malformed);
    } else if (s.voxel_kind == VoxelKind::kDiscrete) {
      std::snprintf(buf, sizeof buf,
                    "%d rows read and none of their indices name a class in this phantom.\n"
                    "  The first column is the value in the segmentation, not the row number.",
                    r.rows);
    } else {
      std::snprintf(buf, sizeof buf,
                    "%d rows read and none of them name a band. This volume is classified by\n"
                    "  value bands, so the first column is the band's position - 0 for the "
                    "first of %d.",
                    r.rows, static_cast<int>(s.voxel_classes.size()));
    }
    note = buf;
    return r;
  }
  std::snprintf(buf, sizeof buf, "%d of %d rows applied", r.applied, r.rows);
  note = buf;
  if (r.unmatched > 0) {
    std::snprintf(buf, sizeof buf, ", %d matched no class in this phantom", r.unmatched);
    note += buf;
  }
  std::snprintf(buf, sizeof buf, "; %d values read as 0-1 and %d as 0-255", r.on_unit, r.on_255);
  note += buf;
  if (r.clamped > 0) {
    std::snprintf(buf, sizeof buf, ", %d clamped to range", r.clamped);
    note += buf;
  }
  if (r.malformed > 0) {
    std::snprintf(buf, sizeof buf, ", %d lines skipped as not numeric", r.malformed);
    note += buf;
  }
  return r;
}

/// Turns a voxel array into material classes on @p s, ready for the user to assign materials.
/// Discrete: one class per distinct value, capped. Continuous: HU bands.
///
/// @param cmap which colormap the classes are coloured from. The default is the hue sweep,
///        which is what this did before there was a choice, so an import that does not ask
///        for a map comes out as it always did.
inline void ClassifyVoxels(const VoxelData& v, VoxelKind kind, Solid& s, std::string& note,
                           Colormap cmap = Colormap::kHueSweep) {
  s.voxel_kind = kind;
  s.voxel_classes.clear();
  char buf[256];

  if (kind == VoxelKind::kDiscrete) {
    // The distinct values, each becoming one material class.
    //
    // MANY CLASSES IS NOT EVIDENCE OF A CONTINUOUS FILE. This used to stop at 64 and refuse
    // with "this looks continuous, not segmented", which is an inference the data does not
    // support and which contradicts the answer the user has already given: kDiscrete is set
    // because they said these are material indices. A segmented phantom with a hundred
    // organs, or an ICRP mesh model, or any lookup table with more entries than 64, was
    // rejected out of hand and the suggested remedy - re-import as Hounsfield units - would
    // have banded indices as if they were densities.
    //
    // What DOES distinguish an index volume is the values themselves: indices are integers.
    // So integrality is what is required here, and the count is only a resource limit.
    //
    // The scan is a direct-address bitmap over the integer range rather than the linear
    // search this had. At 64 classes a linear search costs 64 comparisons a voxel and nobody
    // noticed; raising the cap to 4096 would have made it 4096 comparisons over as many as
    // 1e8 voxels, which is not a slower import, it is one that never finishes.
    constexpr int kMaxClasses = 4096;
    double dlo = 1e300, dhi = -1e300;
    bool integral = true;
    for (float x : v.value) {
      if (x < dlo) { dlo = x; }
      if (x > dhi) { dhi = x; }
      if (x != std::floor(x)) { integral = false; }
    }
    if (v.value.empty()) {
      note = "the file holds no voxels";
      return;
    }
    if (!integral) {
      std::snprintf(buf, sizeof buf,
                    "these values are not whole numbers, so they are not material indices.\n"
                    "  Re-import as continuous (HU) if it is a CT or a density map.");
      note = buf;
      return;
    }
    // Room for the bitmap, not room for the list: a range of a few million costs a few
    // hundred kilobytes and is worth spending to find out how many classes there really are.
    const double span = dhi - dlo + 1.0;
    if (span > 1.0e7) {
      std::snprintf(buf, sizeof buf,
                    "material indices from %g to %g span more than 10 million values.\n"
                    "  That is a range no index table has; re-import as continuous (HU).",
                    dlo, dhi);
      note = buf;
      return;
    }
    const long long lo_i = static_cast<long long>(dlo);
    std::vector<bool> present(static_cast<std::size_t>(span), false);
    for (float x : v.value) {
      present[static_cast<std::size_t>(static_cast<long long>(x) - lo_i)] = true;
    }
    std::vector<float> seen;
    for (std::size_t i = 0; i < present.size(); ++i) {
      if (present[i]) { seen.push_back(static_cast<float>(lo_i + static_cast<long long>(i))); }
    }
    if (static_cast<int>(seen.size()) > kMaxClasses) {
      std::snprintf(buf, sizeof buf,
                    "%d distinct indices, and the limit is %d - not because this looks\n"
                    "  continuous but because a list that long cannot be assigned by hand.",
                    static_cast<int>(seen.size()), kMaxClasses);
      note = buf;
      return;
    }
    // Already ascending: the bitmap was walked in order.
    for (float q : seen) {
      VoxelClass c;
      c.value = q;
      c.value_max = q;
      char lbl[32];
      std::snprintf(lbl, sizeof lbl, "value %g", q);
      c.label = lbl;
      // The colour is filled in after this loop, not here: where a class lands in the
      // colormap depends on the RANGE of the values, and the last value is not known until
      // the loop has finished.
      // EVERY CLASS THE SAME, INCLUDING INDEX 0.
      //
      // Index 0 used to arrive at zero opacity, on the reasoning that in every segmentation
      // convention 0 is "nothing here" - air, background, outside the patient. That is usually
      // true and it is not this code's call to make: the convention is the file's, index 0 is
      // sometimes a real structure, and a class that arrives invisible without being asked for
      // is a class the user has to discover before they can wonder where it went. Anything not
      // wanted in the scene now has a way to say so that means it - the null layer - and it
      // applies to any class, not to whichever one happens to be numbered zero.
      //
      // A tenth for all of them, because the renderer composites front to back and a phantom
      // of opaque cells shows only the first surface a ray meets - which for a segmentation is
      // the skin, and the skin is the one structure nobody imports a phantom to look at. At a
      // tenth, thirty cells of tissue accumulate to about 96% and the interior reads through.
      c.opacity = 0.1f;
      // THE CONTAINER'S MATERIAL, not nothing.
      //
      // A class with no material assigned makes the whole volume unbuildable - build_scene
      // refuses to place a volume whose material is missing - so an import used to arrive in a
      // state where the scene could not be built until every one of what may be two hundred
      // rows had been visited. The volume's own material is the honest default: it is what a
      // cell that matches no class already falls back to, so this makes the classes agree with
      // the rest of the grid rather than inventing a value. Wrong for most classes on a
      // segmentation, and visibly wrong in one place - the material column - rather than
      // invisibly absent.
      c.material = s.material;
      s.voxel_classes.push_back(c);
    }

    // A COLOUR PER CLASS, THE MAP STRETCHED OVER THE RANGE OF THE VALUES.
    //
    // The position in the map is (value - lowest) / (highest - lowest), so the map is fitted
    // to the indices that are actually present rather than being sampled out of a fixed 256
    // and rather than being cycled once it runs out. Neither of those two happens here: no
    // class repeats another's colour, because no two classes have the same value.
    //
    // By VALUE and not by position in the list, which is what "the range of the index values"
    // means and which has a use: two phantoms labelled by the same convention come out with
    // the same colours whether or not both contain every structure. The cost is that a
    // sparse numbering - three classes at 0,1,2 and two at 700,701 - gives each cluster
    // nearly one colour. A colour table is the answer for a phantom numbered like that, and
    // it is chosen in the same dialog.
    {
      const double span = dhi - dlo;
      for (VoxelClass& c : s.voxel_classes) {
        const double t = (span > 0) ? (c.value - dlo) / span : 0.0;
        SampleColormap(cmap, t, c.r, c.g, c.b);
      }
    }
    // The initial state is SPELLED OUT, not just applied. "Index 0 is treated like every other
    // class" has had to be asked for twice, and a log line that names the opacity and says all
    // of them are visible is something the user can check against what they are looking at
    // without reading this file.
    std::snprintf(buf, sizeof buf,
                  "%d distinct values, one material class each on the volume's material, "
                  "coloured by %s; all visible at %.0f%% opacity and on the volume's layer, "
                  "index 0 included",
                  static_cast<int>(s.voxel_classes.size()),
                  ColormapNames()[static_cast<int>(cmap)],
                  100.0 * (s.voxel_classes.empty() ? 0.0 : s.voxel_classes[0].opacity));
    note = buf;
    return;
  }

  // Continuous: the standard Schneider-style HU bands. The boundaries are the ones a CT
  // planning system uses; the materials still have to be assigned by the user, because which
  // tissue substitute belongs in each band is a modelling choice, not a property of the data.
  struct Band { double lo, hi; const char* label; float r, g, b; };
  static const Band kBands[] = {
      {-1050, -950, "air", 0.20f, 0.22f, 0.28f},
      {-950, -700, "lung (inflated)", 0.45f, 0.35f, 0.55f},
      {-700, -100, "lung / fat boundary", 0.55f, 0.45f, 0.40f},
      {-100, -20, "adipose", 0.85f, 0.78f, 0.55f},
      {-20, 20, "water / soft tissue", 0.60f, 0.75f, 0.85f},
      {20, 80, "muscle", 0.80f, 0.40f, 0.40f},
      {80, 300, "trabecular bone", 0.90f, 0.85f, 0.70f},
      {300, 1000, "cortical bone", 0.95f, 0.95f, 0.90f},
      {1000, 3100, "dense bone / implant", 1.00f, 1.00f, 1.00f},
  };
  double lo = 1e30, hi = -1e30;
  for (float x : v.value) {
    lo = std::min<double>(lo, x);
    hi = std::max<double>(hi, x);
  }
  for (const Band& b : kBands) {
    if (b.hi < lo || b.lo > hi) { continue; }  // no voxels in this band
    VoxelClass c;
    c.value = b.lo;
    c.value_max = b.hi;
    c.label = b.label;
    c.r = b.r;
    c.g = b.g;
    c.b = b.b;
    c.material = s.material;   // the container's, as in the discrete case above
    s.voxel_classes.push_back(c);
  }
  std::snprintf(buf, sizeof buf,
                "HU range %.0f to %.0f, %d bands; assign a material to each", lo, hi,
                static_cast<int>(s.voxel_classes.size()));
  note = buf;
}

}  // namespace g4gpu::builder
