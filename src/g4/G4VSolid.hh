// The G4VSolid hierarchy, as a thin builder layer over src/geometry/solids.cuh.
//
// A G4Solid object here is not the thing the GPU sees. It is a description that, when the
// detector is finished, emits itself into a flat pool of geom::Solid records plus the shared
// aux array those records index. That keeps the authoring API object-oriented and virtual, and
// the device representation flat and virtual-free, without either one compromising for the
// other.
//
// Constructor signatures and argument order follow Geant4 exactly, including the places where
// Geant4's order is not the obvious one (G4Trd takes dx1, dx2, dy1, dy2, dz; G4Cons takes the
// two radii of one face before the two of the other). Existing detector code is meant to
// compile unchanged, so the arguments cannot be tidied.
#pragma once
#include <cmath>
#include <memory>
#include <vector>
#include "g4/G4SystemOfUnits.hh"
#include "g4/G4ThreeVector.hh"
#include "g4/G4Types.hh"
#include "geometry/bvh_build.hh"
#include "geometry/solids.cuh"

namespace g4gpu::g4 {

using Sol = geom::Solid<G4double>;
using Xf = geom::Transform<G4double>;

/// The flat pools a finished detector is emitted into.
struct SolidPool {
  std::vector<Sol> solids;
  std::vector<Xf> xforms;
  std::vector<G4double> aux;
  /// Per-cell materials for every voxel volume, concatenated; a solid's `a` is its offset.
  std::vector<short> voxel_cells;
  /// Per-cell CLASS index, the same length and the same offsets as voxel_cells. Render-only:
  /// the transport reads the material, and a phantom whose classes have no materials assigned
  /// yet still has to be drawable - that is the whole point of it, since you assign materials
  /// by looking at the picture. -1 where a cell's value matched no class.
  std::vector<short> voxel_class_cells;
  /// One 0xAARRGGBB per voxel class, concatenated over volumes. A voxel solid's p[6] is where
  /// its own run starts and p[7] how long it is - the same trick as `a` for the cells, and for
  /// the same reason: a flattened volume has no back-reference to the model solid it came
  /// from, so anything the renderer needs per voxel volume has to ride on the solid.
  std::vector<unsigned int> voxel_class_rgba;
  /// Triangles (9 reals each) and BVH nodes (8 reals each) for every mesh in the scene.
  /// G4TessellatedSolid::Build appends to both through geom::build_bvh.
  std::vector<G4double> tri;
  std::vector<G4double> bvh;

  int add_solid(const Sol& s) {
    solids.push_back(s);
    return static_cast<int>(solids.size()) - 1;
  }
  int add_xform(const Xf& x) {
    xforms.push_back(x);
    return static_cast<int>(xforms.size()) - 1;
  }
  int add_aux(const G4double* v, int n) {
    const int off = static_cast<int>(aux.size());
    aux.insert(aux.end(), v, v + n);
    return off;
  }
  /// Appends a volume's cells and returns its offset.
  ///
  /// @p classes may be null, in which case the class array is filled with -1 so that it stays
  /// the same length as the material array. Keeping the two in lockstep is what lets one
  /// offset - the solid's `a` - index both, and VoxelGrid::index already carries it.
  int add_class_colours(const unsigned int* rgba, int n) {
    const int off = static_cast<int>(voxel_class_rgba.size());
    voxel_class_rgba.insert(voxel_class_rgba.end(), rgba, rgba + n);
    return off;
  }
  int add_voxels(const short* cells, int n, const short* classes = nullptr) {
    const int off = static_cast<int>(voxel_cells.size());
    voxel_cells.insert(voxel_cells.end(), cells, cells + n);
    if (classes != nullptr) {
      voxel_class_cells.insert(voxel_class_cells.end(), classes, classes + n);
    } else {
      voxel_class_cells.insert(voxel_class_cells.end(), static_cast<std::size_t>(n),
                               static_cast<short>(-1));
    }
    return off;
  }
  geom::SolidStore<G4double> store() const {
    geom::SolidStore<G4double> st{};
    st.solids = solids.empty() ? nullptr : solids.data();
    st.xforms = xforms.empty() ? nullptr : xforms.data();
    st.aux = aux.empty() ? nullptr : aux.data();
    st.tri = tri.empty() ? nullptr : tri.data();
    st.bvh = bvh.empty() ? nullptr : bvh.data();
    return st;
  }
};

}  // namespace g4gpu::g4

/// Base of every solid. `Build` appends this solid to @p pool and returns its index there.
class G4VSolid {
 public:
  explicit G4VSolid(const G4String& name) : name_(name) {}
  virtual ~G4VSolid() = default;
  G4VSolid(const G4VSolid&) = delete;
  G4VSolid& operator=(const G4VSolid&) = delete;

  const G4String& GetName() const { return name_; }
  void SetName(const G4String& n) { name_ = n; }

  virtual G4int Build(g4gpu::g4::SolidPool& pool) const = 0;

  /// Rough half-extent, used to place a default camera and to size the world when the user
  /// does not give one. Not a physics quantity; an overestimate is harmless.
  virtual G4double Extent() const = 0;

 protected:
  static g4gpu::g4::Sol make(g4gpu::geom::SolidType t, std::initializer_list<G4double> ps) {
    g4gpu::g4::Sol s{};
    s.type = t;
    int i = 0;
    for (G4double v : ps) {
      if (i < 8) { s.p[i++] = v; }
    }
    s.xform = -1;
    return s;
  }

 private:
  G4String name_;
};
