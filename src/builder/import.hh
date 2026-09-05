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

/// Turns a voxel array into material classes on `solid`, ready for the user to assign
/// materials. Discrete: one class per distinct value, capped. Continuous: HU bands.
void ClassifyVoxels(const VoxelData& v, VoxelKind kind, Solid& solid, std::string& note);

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

inline void ClassifyVoxels(const VoxelData& v, VoxelKind kind, Solid& s, std::string& note) {
  s.voxel_kind = kind;
  s.voxel_classes.clear();
  char buf[256];

  if (kind == VoxelKind::kDiscrete) {
    // Distinct values, capped: a file that turns out to be continuous would otherwise produce
    // tens of thousands of classes and an unusable list.
    constexpr int kMaxClasses = 64;
    std::vector<float> seen;
    for (float x : v.value) {
      bool have = false;
      for (float q : seen) {
        if (q == x) { have = true; break; }
      }
      if (!have) {
        seen.push_back(x);
        if (static_cast<int>(seen.size()) > kMaxClasses) { break; }
      }
    }
    if (static_cast<int>(seen.size()) > kMaxClasses) {
      std::snprintf(buf, sizeof buf,
                    "more than %d distinct values: this looks continuous, not segmented.\n"
                    "  Re-import as continuous (HU) if it is a CT.",
                    kMaxClasses);
      note = buf;
      return;
    }
    std::sort(seen.begin(), seen.end());
    for (float q : seen) {
      VoxelClass c;
      c.value = q;
      c.value_max = q;
      char lbl[32];
      std::snprintf(lbl, sizeof lbl, "value %g", q);
      c.label = lbl;
      // A distinguishable colour per class, spread round the hue circle.
      const double h = 360.0 * s.voxel_classes.size() / std::max<std::size_t>(1, seen.size());
      const double hh = h / 60.0;
      const int i = static_cast<int>(hh) % 6;
      const double fr = hh - static_cast<int>(hh);
      const double p = 0.35, qv = 1.0 - 0.65 * fr, t = 0.35 + 0.65 * fr;
      switch (i) {
        case 0: c.r = 1.0f; c.g = static_cast<float>(t); c.b = static_cast<float>(p); break;
        case 1: c.r = static_cast<float>(qv); c.g = 1.0f; c.b = static_cast<float>(p); break;
        case 2: c.r = static_cast<float>(p); c.g = 1.0f; c.b = static_cast<float>(t); break;
        case 3: c.r = static_cast<float>(p); c.g = static_cast<float>(qv); c.b = 1.0f; break;
        case 4: c.r = static_cast<float>(t); c.g = static_cast<float>(p); c.b = 1.0f; break;
        default: c.r = 1.0f; c.g = static_cast<float>(p); c.b = static_cast<float>(qv); break;
      }
      s.voxel_classes.push_back(c);
    }
    std::snprintf(buf, sizeof buf, "%d distinct values, one material class each",
                  static_cast<int>(s.voxel_classes.size()));
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
    s.voxel_classes.push_back(c);
  }
  std::snprintf(buf, sizeof buf,
                "HU range %.0f to %.0f, %d bands; assign a material to each", lo, hi,
                static_cast<int>(s.voxel_classes.size()));
  note = buf;
}

}  // namespace g4gpu::builder
