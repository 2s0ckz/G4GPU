// Writes a Model out as a compilable Geant4-style project.
//
// This is the point of the Save command. The user does not get a proprietary file that only
// this GUI can open; they get exactly what examples/B1 is - exampleX.cc, include/, src/, a
// build script and a macro - which they can read, edit by hand, put under version control,
// and compile without the GUI ever running again.
//
// The generated code is therefore written to be *read*: real names, the same call order a
// person would use, units spelled out (5*cm, not 50), and a comment where the model made a
// choice that is not obvious from the numbers.
//
// A .model file is written alongside it so the GUI can reopen the document with its editing
// state intact. That file is a convenience; the C++ is the artefact.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "builder/model.hh"

namespace g4gpu::builder {

namespace detail {

/// Formats a length in mm as the neatest exact expression a reader would write.
inline std::string Len(double mm_value) {
  char buf[64];
  if (mm_value == 0) { return "0"; }
  const double in_m = mm_value / 1000.0;
  const double in_cm = mm_value / 10.0;
  // Prefer the largest unit that leaves a short decimal, because "0.5*m" reads better than
  // "500*mm" and both are exact.
  if (in_m == static_cast<long long>(in_m * 1000) / 1000.0 && std::fabs(in_m) >= 1.0) {
    std::snprintf(buf, sizeof buf, "%g*m", in_m);
  } else if (in_cm == static_cast<long long>(in_cm * 1000) / 1000.0
             && std::fabs(in_cm) >= 1.0) {
    std::snprintf(buf, sizeof buf, "%g*cm", in_cm);
  } else {
    std::snprintf(buf, sizeof buf, "%g*mm", mm_value);
  }
  return buf;
}

inline std::string Deg(double d) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g*deg", d);
  return buf;
}

inline std::string Num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

/// A C++ identifier from a user-supplied name.
inline std::string Ident(const std::string& s, const char* prefix = "") {
  std::string out = prefix;
  bool upper = (prefix[0] != '\0');
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      out.push_back(upper ? static_cast<char>(std::toupper(c)) : c);
      upper = false;
    } else {
      upper = true;  // a separator capitalises the next letter
    }
  }
  if (out.empty()) { out = "X"; }
  if (std::isdigit(static_cast<unsigned char>(out[0]))) { out = "_" + out; }
  return out;
}

inline std::string SolidVar(const Model& m, int i) {
  return Ident(m.solids[i].name, "solid");
}

/// A brace list of the solid's (z, rInner, rOuter) triples, in Geant4 length units.
///
/// Kept as one flat list rather than three, so the generated line reads in the same order as
/// the builder's own section editor and a reader can see the triples.
inline std::string Sections(const Solid& s) {
  std::string out = "{";
  for (std::size_t i = 0; i + 2 < s.sections.size(); i += 3) {
    if (i != 0) { out += ", "; }
    out += Len(s.sections[i]) + ", " + Len(s.sections[i + 1]) + ", " + Len(s.sections[i + 2]);
  }
  return out + "}";
}

inline std::string LogicVar(const Model& m, int i) {
  return Ident(m.solids[i].name, "logic");
}
inline std::string MatVar(const Model& m, int i) {
  return Ident(m.materials[i].name, "mat");
}
inline std::string ElemVar(const Model& m, int i) {
  return Ident(m.elements[i].name, "el");
}

/// The `new G4Xxx(...)` expression for one solid.
inline std::string SolidExpr(const Model& m, int i) {
  const Solid& s = m.solids[i];
  const std::string q = "\"" + s.name + "\"";
  const double* p = s.p;
  auto L = [&](int k) { return Len(p[k]); };
  auto A = [&](int k) { return Deg(p[k]); };
  switch (s.shape) {
    case Shape::kBox:
      return "new G4Box(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ")";
    case Shape::kTubs:
      return "new G4Tubs(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + A(3) + ", "
             + A(4) + ")";
    case Shape::kCons:
      return "new G4Cons(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + L(3) + ", "
             + L(4) + ", " + A(5) + ", " + A(6) + ")";
    case Shape::kSphere:
      return "new G4Sphere(" + q + ", " + L(0) + ", " + L(1) + ", " + A(2) + ", " + A(3)
             + ", " + A(4) + ", " + A(5) + ")";
    case Shape::kOrb:
      return "new G4Orb(" + q + ", " + L(0) + ")";
    case Shape::kTorus:
      return "new G4Torus(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + A(3) + ", "
             + A(4) + ")";
    case Shape::kTrd:
      return "new G4Trd(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + L(3) + ", "
             + L(4) + ")";
    case Shape::kTrap:
      return "new G4Trap(" + q + ", " + L(0) + ", " + A(1) + ", " + A(2) + ", " + L(3) + ", "
             + L(4) + ", " + L(5) + ", " + A(6) + ", " + L(7) + ", " + L(8) + ", " + L(9)
             + ", " + A(10) + ")";
    case Shape::kPara:
      return "new G4Para(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + A(3) + ", "
             + A(4) + ", " + A(5) + ")";
    case Shape::kEllipticalTube:
      return "new G4EllipticalTube(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ")";
    case Shape::kEllipsoid:
      return "new G4Ellipsoid(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", " + L(3)
             + ", " + L(4) + ")";
    case Shape::kEllipticalCone:
      return "new G4EllipticalCone(" + q + ", " + Num(p[0]) + ", " + Num(p[1]) + ", " + L(2)
             + ", " + L(3) + ")";
    case Shape::kParaboloid:
      return "new G4Paraboloid(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ")";
    case Shape::kHype:
      return "new G4Hype(" + q + ", " + L(0) + ", " + L(1) + ", " + A(2) + ", " + A(3) + ", "
             + L(4) + ")";
    case Shape::kTet:
      return "new G4Tet(" + q + ", G4ThreeVector(" + L(0) + ", " + L(1) + ", " + L(2)
             + "), G4ThreeVector(" + L(3) + ", " + L(4) + ", " + L(5) + "), G4ThreeVector("
             + L(6) + ", " + L(7) + ", " + L(8) + "), G4ThreeVector(" + L(9) + ", " + L(10)
             + ", " + L(11) + "))";
    case Shape::kUnion:
    case Shape::kSubtraction:
    case Shape::kIntersection: {
      const char* op = (s.shape == Shape::kUnion) ? "G4UnionSolid"
                       : (s.shape == Shape::kSubtraction) ? "G4SubtractionSolid"
                                                          : "G4IntersectionSolid";
      const std::string a = (s.operand_a >= 0) ? SolidVar(m, s.operand_a) : "nullptr";
      const std::string b = (s.operand_b >= 0) ? SolidVar(m, s.operand_b) : "nullptr";
      // The offset is the second operand's placement relative to the first.
      std::string off = "G4ThreeVector()";
      if (s.operand_a >= 0 && s.operand_b >= 0) {
        const Solid& sa = m.solids[s.operand_a];
        const Solid& sb = m.solids[s.operand_b];
        off = "G4ThreeVector(" + Len(sb.pos[0] - sa.pos[0]) + ", "
              + Len(sb.pos[1] - sa.pos[1]) + ", " + Len(sb.pos[2] - sa.pos[2]) + ")";
      }
      return std::string("new ") + op + "(" + q + ", " + a + ", " + b + ", nullptr, " + off
             + ")";
    }
    case Shape::kImportedMesh:
      // The triangles are in a sidecar, as the voxel cells are: a coarse CAD import is tens
      // of thousands of triangles, which is 300,000 numbers no compiler should be asked to
      // parse and nobody should be asked to read.
      return "new G4TessellatedSolid(" + q + ")";
    case Shape::kVoxelGrid:
      // The cells themselves are written to a sidecar file and loaded at construction: a
      // 512^3 grid is 134 million initialisers, which no compiler will accept and no one
      // would want to read. The generated code says where the data went.
      return "new G4VoxelGrid(" + q + ", " + L(0) + ", " + L(1) + ", " + L(2) + ", "
             + Num(p[3]) + ", " + Num(p[4]) + ", " + Num(p[5]) + ")";
    case Shape::kPolycone:
      // The (z, rInner, rOuter) sections go through a generated helper rather than three
      // arrays, because EmitSolid returns one expression and G4Polycone's constructor wants
      // three pointers. The helper splits the flat triple list; see WritePolyHelper.
      return "MakePolycone(" + q + ", " + A(0) + ", " + A(1) + ", " + Sections(s) + ")";
    case Shape::kPolyhedra:
      return "MakePolyhedra(" + q + ", " + A(0) + ", " + A(1) + ", " + Num(p[2]) + ", "
             + Sections(s) + ")";
    default:
      // Not an expression that fails to compile. A polycone used to land here and the
      // generated project came out with
      //
      //     auto* solidCone = nullptr /* Polycone is not yet emitted */;
      //
      // which is a compile error in a file the user is told is a working project - and one
      // nothing noticed, because no automated run had ever saved a model containing one.
      // Whatever cannot be emitted is now refused by CanEmitSolid before a file is written.
      return "nullptr /* " + std::string(ShapeName(s.shape))
             + ": unreachable, CanEmitSolid refuses this shape */";
  }
}


/// True when the model has a polycone or a polyhedra, and therefore needs the section helper.
inline bool NeedsPolyHelper(const Model& m) {
  for (const Solid& s : m.solids) {
    if (s.shape == Shape::kPolycone || s.shape == Shape::kPolyhedra) { return true; }
  }
  return false;
}

/// Whether this shape can be written as C++ at all. Anything false here is refused when the
/// project is saved, with the shape named, rather than emitted as code that does not compile.
inline bool CanEmitSolid(const Solid& s) {
  switch (s.shape) {
    case Shape::kPolycone:
    case Shape::kPolyhedra:
      // Three sections minimum, and the list has to be whole triples: G4Polycone indexes
      // three parallel arrays and a short one reads off the end.
      return s.sections.size() >= 9 && (s.sections.size() % 3) == 0;
    default:
      return true;
  }
}

inline bool NeedsVoxelLoader(const Model& m) {
  for (const Solid& s : m.solids) {
    if (s.shape == Shape::kVoxelGrid) { return true; }
  }
  return false;
}

/// True when the model has an imported mesh, and therefore needs the triangle loader.
inline bool NeedsMeshLoader(const Model& m) {
  for (const Solid& s : m.solids) {
    if (s.shape == Shape::kImportedMesh) { return true; }
  }
  return false;
}
}  // namespace detail

/// Writes the project into `dir`. Returns false with a message printed on failure.
bool WriteProject(const Model& model, const std::string& dir);

/// Saves the editable document. Reloadable by ReadModel.
bool WriteModelFile(const Model& model, const std::string& path);
bool ReadModelFile(Model& model, const std::string& path);

}  // namespace g4gpu::builder
