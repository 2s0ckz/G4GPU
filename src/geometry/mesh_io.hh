// Reading triangle meshes: STL (binary and ASCII), Wavefront OBJ, and ASCII PLY.
//
// The same three formats christopherpoole/cadmesh reads without its optional ASSIMP
// dependency, which is not a coincidence: CADMesh.hh presents cadmesh's interface over this,
// so code written against that library compiles here.
//
// Everything is triangulated on the way in. An OBJ or PLY face with more than three vertices
// is fanned from its first vertex, which is correct for a convex face and is what every mesh
// reader does; a concave quad is rare in exported CAD and would need ear clipping.
//
// One design point worth stating: a mesh file carries no units. STL is conventionally
// millimetres and OBJ conventionally is not, so a scale factor is part of the read rather than
// something applied afterwards, and the note the reader fills in reports the bounding box in
// whatever units came out. A model that arrives 1000x too large is the most common import
// failure there is, and the bounding box in the note is what makes it obvious.
#pragma once
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace g4gpu::geom {

/// Nine reals per triangle, three vertices each.
using MeshTriangles = std::vector<double>;

namespace mesh_io_detail {

inline bool HasSuffix(const std::string& s, const char* suffix) {
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

/// Appends the triangle fan of a polygon given by vertex indices into @p v (3 reals each).
inline void FanFace(const std::vector<double>& v, const std::vector<int>& idx,
                    MeshTriangles& out) {
  for (std::size_t k = 2; k < idx.size(); ++k) {
    const int a = idx[0], b = idx[k - 1], c = idx[k];
    const int n = static_cast<int>(v.size() / 3);
    if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) { continue; }
    for (int j = 0; j < 3; ++j) { out.push_back(v[static_cast<std::size_t>(a) * 3 + j]); }
    for (int j = 0; j < 3; ++j) { out.push_back(v[static_cast<std::size_t>(b) * 3 + j]); }
    for (int j = 0; j < 3; ++j) { out.push_back(v[static_cast<std::size_t>(c) * 3 + j]); }
  }
}

inline void Describe(const char* what, const MeshTriangles& tri, std::string& note) {
  char buf[256];
  if (tri.empty()) {
    std::snprintf(buf, sizeof buf, "%s, no triangles", what);
    note = buf;
    return;
  }
  double lo[3] = {tri[0], tri[1], tri[2]};
  double hi[3] = {tri[0], tri[1], tri[2]};
  for (std::size_t i = 0; i + 2 < tri.size(); i += 3) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = (tri[i + k] < lo[k]) ? tri[i + k] : lo[k];
      hi[k] = (tri[i + k] > hi[k]) ? tri[i + k] : hi[k];
    }
  }
  std::snprintf(buf, sizeof buf,
                "%s, %zu triangles, bounds %.3g..%.3g x %.3g..%.3g x %.3g..%.3g", what,
                tri.size() / 9, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
  note = buf;
}

}  // namespace mesh_io_detail

/// STL, binary or ASCII.
inline bool ReadStlMesh(const std::string& path, MeshTriangles& tri, std::string& note) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    note = "cannot open " + path;
    return false;
  }
  char head[84] = {0};
  const std::size_t got = std::fread(head, 1, 84, f);
  if (got < 15) {
    std::fclose(f);
    note = path + ": too short for an STL";
    return false;
  }

  // A binary STL's size is exactly 84 + 50 * count. Testing that, rather than trusting the
  // "solid" banner, is the reliable discriminator: plenty of binary STLs begin with "solid".
  std::fseek(f, 0, SEEK_END);
  const long long bytes = std::ftell(f);
  std::uint32_t count = 0;
  std::memcpy(&count, head + 80, 4);
  const bool binary = (bytes == 84LL + 50LL * count) && count > 0;

  tri.clear();
  if (binary) {
    std::fseek(f, 84, SEEK_SET);
    tri.reserve(static_cast<std::size_t>(count) * 9);
    for (std::uint32_t i = 0; i < count; ++i) {
      float rec[12];
      std::uint16_t attr = 0;
      if (std::fread(rec, 4, 12, f) != 12) { break; }
      if (std::fread(&attr, 2, 1, f) != 1) { break; }
      // rec[0..2] is the facet normal, which is redundant with the winding and often wrong.
      for (int k = 3; k < 12; ++k) { tri.push_back(static_cast<double>(rec[k])); }
    }
    std::fclose(f);
    mesh_io_detail::Describe("binary STL", tri, note);
    return !tri.empty();
  }

  std::fseek(f, 0, SEEK_SET);
  char line[512];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    const char* v = std::strstr(line, "vertex");
    if (v == nullptr) { continue; }
    double x = 0, y = 0, z = 0;
    if (std::sscanf(v + 6, "%lf %lf %lf", &x, &y, &z) == 3) {
      tri.push_back(x);
      tri.push_back(y);
      tri.push_back(z);
    }
  }
  std::fclose(f);
  if (tri.size() % 9 != 0) {
    const std::size_t drop = tri.size() % 9;
    tri.resize(tri.size() - drop);
    note = path + ": vertex count was not a whole number of triangles; trailing ones dropped";
  }
  mesh_io_detail::Describe("ASCII STL", tri, note);
  return !tri.empty();
}

/// Wavefront OBJ. `v` lines and `f` lines; groups, materials, normals and texture coordinates
/// are skipped, and negative (relative) face indices are resolved.

/// Wavefront OBJ. Triangles and n-gons, fanned; only the vertex indices are read.
inline bool ReadObjMesh(const std::string& path, MeshTriangles& tri, std::string& note) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    note = "cannot open " + path;
    return false;
  }
  std::vector<double> verts;
  tri.clear();
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
      double x = 0, y = 0, z = 0;
      if (std::sscanf(line + 1, "%lf %lf %lf", &x, &y, &z) == 3) {
        verts.push_back(x);
        verts.push_back(y);
        verts.push_back(z);
      }
      continue;
    }
    if (line[0] != 'f' || (line[1] != ' ' && line[1] != '\t')) { continue; }
    // f v, f v/vt, f v//vn, f v/vt/vn - only the vertex index matters here.
    std::vector<int> idx;
    const char* p = line + 1;
    while (*p != 0) {
      while (*p == ' ' || *p == '\t') { ++p; }
      if (*p == 0 || *p == '\n' || *p == '\r') { break; }
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      const int total = static_cast<int>(verts.size() / 3);
      idx.push_back((v > 0) ? static_cast<int>(v - 1) : total + static_cast<int>(v));
      p = end;
      while (*p != 0 && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { ++p; }
    }
    mesh_io_detail::FanFace(verts, idx, tri);
  }
  std::fclose(f);
  mesh_io_detail::Describe("Wavefront OBJ", tri, note);
  return !tri.empty();
}

/// ASCII PLY. Binary PLY is refused with a message rather than misread: the header would parse
/// and the vertex block would come out as noise.
inline bool ReadPlyMesh(const std::string& path, MeshTriangles& tri, std::string& note) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    note = "cannot open " + path;
    return false;
  }
  char line[1024];
  int n_vert = 0, n_face = 0;
  bool ascii = false;
  int vertex_props = 0;
  bool in_vertex_element = false;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    if (std::strncmp(line, "format ", 7) == 0) {
      ascii = (std::strstr(line, "ascii") != nullptr);
    } else if (std::sscanf(line, "element vertex %d", &n_vert) == 1) {
      in_vertex_element = true;
      vertex_props = 0;
    } else if (std::sscanf(line, "element face %d", &n_face) == 1) {
      in_vertex_element = false;
    } else if (std::strncmp(line, "property ", 9) == 0) {
      if (in_vertex_element) { ++vertex_props; }
    } else if (std::strncmp(line, "end_header", 10) == 0) {
      break;
    }
  }
  if (!ascii) {
    std::fclose(f);
    note = path
           + ": binary PLY is not read here. Convert to ASCII PLY, or to STL, and re-import.";
    return false;
  }

  std::vector<double> verts;
  verts.reserve(static_cast<std::size_t>(n_vert) * 3);
  for (int i = 0; i < n_vert; ++i) {
    if (std::fgets(line, sizeof line, f) == nullptr) { break; }
    double x = 0, y = 0, z = 0;
    if (std::sscanf(line, "%lf %lf %lf", &x, &y, &z) == 3) {
      verts.push_back(x);
      verts.push_back(y);
      verts.push_back(z);
    }
  }
  (void)vertex_props;  // x, y, z are always the first three; the rest are colour and normals.

  tri.clear();
  for (int i = 0; i < n_face; ++i) {
    if (std::fgets(line, sizeof line, f) == nullptr) { break; }
    const char* p = line;
    char* end = nullptr;
    const long k = std::strtol(p, &end, 10);
    if (end == p || k < 3) { continue; }
    std::vector<int> idx;
    p = end;
    for (long j = 0; j < k; ++j) {
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      idx.push_back(static_cast<int>(v));
      p = end;
    }
    mesh_io_detail::FanFace(verts, idx, tri);
  }
  std::fclose(f);
  mesh_io_detail::Describe("ASCII PLY", tri, note);
  return !tri.empty();
}

/// Reads whichever mesh format the extension names.
inline bool ReadMeshFile(const std::string& path, MeshTriangles& tri, std::string& note) {
  if (mesh_io_detail::HasSuffix(path, ".stl")) { return ReadStlMesh(path, tri, note); }
  if (mesh_io_detail::HasSuffix(path, ".obj")) { return ReadObjMesh(path, tri, note); }
  if (mesh_io_detail::HasSuffix(path, ".ply")) { return ReadPlyMesh(path, tri, note); }
  note = path + ": unrecognized mesh extension. STL, OBJ and ASCII PLY are read.";
  return false;
}

/// Scales and offsets a mesh in place, which is how a file in the wrong units is brought into
/// millimetres and how CADMesh's SetScale and SetOffset are applied.
inline void TransformMesh(MeshTriangles& tri, double scale, double ox, double oy, double oz) {
  for (std::size_t i = 0; i + 2 < tri.size(); i += 3) {
    tri[i + 0] = tri[i + 0] * scale + ox;
    tri[i + 1] = tri[i + 1] * scale + oy;
    tri[i + 2] = tri[i + 2] * scale + oz;
  }
}

}  // namespace g4gpu::geom
