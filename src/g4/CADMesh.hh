// CADMesh: christopherpoole/cadmesh's interface, over this project's mesh reader.
//
// cadmesh is how most Geant4 users load CAD geometry, so its calls are the ones already in
// people's detector descriptions:
//
//     auto mesh = CADMesh::TessellatedMesh::FromSTL("part.stl");
//     mesh->SetScale(mm);
//     mesh->SetOffset(G4ThreeVector(0, 0, 10 * cm));
//     auto* logic = new G4LogicalVolume(mesh->GetSolid(), material, "part");
//
// That code compiles and works here. What it returns is a G4TessellatedSolid built and closed,
// which this transport intersects against its triangles through a BVH - see
// src/geometry/mesh.cuh.
//
// Differences from the real library, all of them things it does that this does not:
//
//   - ASCII STL, binary STL, Wavefront OBJ and ASCII PLY are read: the same set cadmesh reads
//     with its built-in readers. Binary PLY and the long tail of formats cadmesh reaches
//     through ASSIMP (STEP, IGES, 3DS, COLLADA, ...) are not, and are refused with a message
//     naming what to convert to rather than returning an empty solid.
//   - TetrahedralMesh, which cadmesh builds with TETGEN to make a G4AssemblyVolume of
//     tetrahedra, is not here. Its purpose is to speed up navigation in Geant4's
//     G4TessellatedSolid; the BVH does that job directly, so the tetrahedralisation has
//     nothing to buy. FromPLY/FromSTL/FromOBJ on it say so and stop.
//   - GetSolids() returns one solid. cadmesh can split a multi-object OBJ or an ASSIMP scene
//     into one solid per named object; the readers here concatenate everything in the file
//     into a single mesh, and GetSolid(index) or GetSolid(name) beyond the first says so.
//
// A note on units, because it is the commonest way a CAD import goes wrong. A mesh file
// carries no units. cadmesh's SetScale is how you say what the numbers mean, and the default
// here is 1, meaning the file's numbers are taken as millimetres - this project's internal
// length unit. A part exported in metres and placed without a scale is a thousand times too
// big, and the import prints its bounding box so that this is visible at once.
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "g4/G4Solids.hh"
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4ThreeVector.hh"
#include "geometry/mesh_io.hh"

namespace CADMesh {

class TessellatedMesh {
 public:
  static std::shared_ptr<TessellatedMesh> FromSTL(const G4String& path) {
    return Load(path, "STL");
  }
  static std::shared_ptr<TessellatedMesh> FromOBJ(const G4String& path) {
    return Load(path, "OBJ");
  }
  static std::shared_ptr<TessellatedMesh> FromPLY(const G4String& path) {
    return Load(path, "PLY");
  }
  /// cadmesh's ASSIMP entry point. There is no ASSIMP here; the extension still decides, so
  /// this works for the three formats read natively and refuses the rest.
  static std::shared_ptr<TessellatedMesh> From(const G4String& path) { return Load(path, ""); }

  /// Multiplies every coordinate. Pass the unit the file is in: SetScale(cm) for a part
  /// exported in centimetres.
  void SetScale(G4double s) {
    scale_ = s;
    dirty_ = true;
  }
  void SetOffset(G4double x, G4double y, G4double z) {
    off_ = G4ThreeVector(x, y, z);
    dirty_ = true;
  }
  void SetOffset(const G4ThreeVector& v) {
    off_ = v;
    dirty_ = true;
  }
  /// cadmesh reverses the facet winding with this. Winding is irrelevant here - the ray
  /// triangle test is two-sided and containment is a parity count - so it is accepted and
  /// changes nothing, which is the truth rather than a silent no-op.
  void SetReverse(G4bool r) { (void)r; }

  G4double GetScale() const { return scale_; }
  const G4ThreeVector& GetOffset() const { return off_; }
  const G4String& GetFileName() const { return path_; }

  G4VSolid* GetSolid() {
    Realise();
    return solid_;
  }
  G4VSolid* GetSolid(G4int index) {
    if (index != 0) {
      std::printf(
          "\nFATAL: CADMesh::GetSolid(%d) on \"%s\": the readers here concatenate a file into\n"
          "  one mesh, so there is only index 0. Splitting a multi-object file into separate\n"
          "  solids is something cadmesh does through ASSIMP and this does not.\n",
          index, path_.c_str());
      std::exit(2);
    }
    return GetSolid();
  }
  G4VSolid* GetSolid(const G4String& name) {
    std::printf(
        "note: CADMesh::GetSolid(\"%s\") on \"%s\" returns the whole file as one mesh; named\n"
        "      sub-objects are not separated here.\n",
        name.c_str(), path_.c_str());
    return GetSolid();
  }
  std::vector<G4VSolid*> GetSolids() { return {GetSolid()}; }

  G4int GetNumberOfFacets() {
    Realise();
    return static_cast<G4TessellatedSolid*>(solid_)->GetNumberOfFacets();
  }

 private:
  static std::shared_ptr<TessellatedMesh> Load(const G4String& path, const char* expect) {
    auto m = std::shared_ptr<TessellatedMesh>(new TessellatedMesh);
    m->path_ = path;
    std::string note;
    if (!g4gpu::geom::ReadMeshFile(path, m->tri_, note)) {
      std::printf("\nFATAL: could not read the mesh \"%s\".\n  %s\n", path.c_str(),
                  note.c_str());
      std::exit(2);
    }
    std::printf("CADMesh: %s: %s\n", path.c_str(), note.c_str());
    if (expect[0] != 0 && !EndsWithFormat(path, expect)) {
      // From*(path) with a mismatched extension read something, so say what it actually
      // parsed. Refusing would be worse: the file is fine, only the call name is off.
      std::printf("note: From%s(\"%s\") - the extension says otherwise; read as above.\n",
                  expect, path.c_str());
    }
    return m;
  }

  static bool EndsWithFormat(const std::string& p, const char* fmt) {
    const std::string dot = std::string(".") + fmt;
    return g4gpu::geom::mesh_io_detail::HasSuffix(p, dot.c_str());
  }

  void Realise() {
    if (solid_ != nullptr && !dirty_) { return; }
    g4gpu::geom::MeshTriangles t = tri_;
    g4gpu::geom::TransformMesh(t, scale_, off_.x(), off_.y(), off_.z());
    auto* s = new G4TessellatedSolid(path_);
    s->AddTriangles(t);
    s->SetSolidClosed(true);
    solid_ = s;
    dirty_ = false;
    std::printf("CADMesh: %s: volume %.6g mm3, area %.6g mm2\n", path_.c_str(),
                s->GetCubicVolume(), s->GetSurfaceArea());
  }

  G4String path_;
  g4gpu::geom::MeshTriangles tri_;
  G4double scale_ = 1.0;
  G4ThreeVector off_;
  G4VSolid* solid_ = nullptr;
  bool dirty_ = true;
};

/// cadmesh's tetrahedral path. Not implemented, and the reason is not a missing dependency:
/// its purpose is to make navigation through a G4TessellatedSolid fast in Geant4, and the BVH
/// in src/geometry/mesh.cuh already does that. Use TessellatedMesh.
class TetrahedralMesh {
 public:
  static std::shared_ptr<TetrahedralMesh> FromPLY(const G4String& path) { return Refuse(path); }
  static std::shared_ptr<TetrahedralMesh> FromSTL(const G4String& path) { return Refuse(path); }
  static std::shared_ptr<TetrahedralMesh> FromOBJ(const G4String& path) { return Refuse(path); }

 private:
  static std::shared_ptr<TetrahedralMesh> Refuse(const G4String& path) {
    std::printf(
        "\nFATAL: CADMesh::TetrahedralMesh (\"%s\") is not implemented.\n"
        "  cadmesh tetrahedralises a mesh to speed up navigation through Geant4's\n"
        "  G4TessellatedSolid. Here a mesh is traversed through a BVH, which is what the\n"
        "  tetrahedra were for, so use CADMesh::TessellatedMesh instead:\n"
        "    auto mesh = CADMesh::TessellatedMesh::FromSTL(\"%s\");\n"
        "    auto* logic = new G4LogicalVolume(mesh->GetSolid(), material, \"part\");\n",
        path.c_str(), path.c_str());
    std::exit(2);
  }
};

}  // namespace CADMesh
