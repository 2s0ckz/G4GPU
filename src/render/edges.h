// The wireframe edge list, built on the host and drawn by vis::render_edges.
//
// SHARED BY THE VIEWER AND THE BUILDER, and it was not: this lived inside vis_manager.cu, so
// the builder had no way to draw an edge and never launched the pass. Its own styles said
// `solid = visible && !wireframe`, which is right - a wireframe volume must not be ray cast as
// a surface - so a volume set to wireframe there was simply invisible, and the default world,
// which arrives with wireframe on, had no outline at all. Reported as wireframe rendering not
// working, which is exactly what it was.
//
// One implementation rather than two, for the reason that keeps coming up in this project: a
// second copy is a second thing to fix, and the one nobody remembers is the one that is wrong.

#pragma once

#include <cstddef>
#include <vector>

#include "geometry/navigator.cuh"
#include "geometry/transform.cuh"
#include "core/vec3.cuh"

namespace g4gpu::vis {

/// Line segments in world coordinates, one colour each, as vis::render_edges wants them: six
/// parallel float arrays rather than an array of structs, so the pass reads them coalesced.
struct EdgeList {
  std::vector<float> x0, y0, z0, x1, y1, z1;
  std::vector<unsigned int> rgb;

  void Add(float ax, float ay, float az, float bx, float by, float bz, unsigned int c) {
    x0.push_back(ax); y0.push_back(ay); z0.push_back(az);
    x1.push_back(bx); y1.push_back(by); z1.push_back(bz);
    rgb.push_back(c);
  }

  /// The twelve edges of an axis-aligned box, transformed by a volume's placement.
  template <typename real_t>
  void AddBox(const geom::Transform<real_t>& xf, real_t hx, real_t hy, real_t hz,
              unsigned int c) {
    const real_t sx[2] = {-hx, hx}, sy[2] = {-hy, hy}, sz[2] = {-hz, hz};
    auto pt = [&](real_t x, real_t y, real_t z) {
      // Local to world: the stored matrix is world -> local, so the inverse is its transpose.
      const Vec3<real_t> l{x, y, z};
      const real_t* r = xf.rot;
      return Vec3<real_t>{r[0] * l.x + r[3] * l.y + r[6] * l.z + xf.trans.x,
                          r[1] * l.x + r[4] * l.y + r[7] * l.z + xf.trans.y,
                          r[2] * l.x + r[5] * l.y + r[8] * l.z + xf.trans.z};
    };
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        const auto a1 = pt(sx[0], sy[i], sz[j]), b1 = pt(sx[1], sy[i], sz[j]);
        const auto a2 = pt(sx[i], sy[0], sz[j]), b2 = pt(sx[i], sy[1], sz[j]);
        const auto a3 = pt(sx[i], sy[j], sz[0]), b3 = pt(sx[i], sy[j], sz[1]);
        Add(static_cast<float>(a1.x), static_cast<float>(a1.y), static_cast<float>(a1.z),
            static_cast<float>(b1.x), static_cast<float>(b1.y), static_cast<float>(b1.z), c);
        Add(static_cast<float>(a2.x), static_cast<float>(a2.y), static_cast<float>(a2.z),
            static_cast<float>(b2.x), static_cast<float>(b2.y), static_cast<float>(b2.z), c);
        Add(static_cast<float>(a3.x), static_cast<float>(a3.y), static_cast<float>(a3.z),
            static_cast<float>(b3.x), static_cast<float>(b3.y), static_cast<float>(b3.z), c);
      }
    }
  }

  /// The edges a volume contributes, or none.
  ///
  /// BOXES ONLY, and their real half-lengths. A bounding cube drawn round a cone or a sphere is
  /// not a hint about its shape, it is a lie about it - and the curved solids are ray cast as
  /// surfaces anyway, so they need no outline. A voxel grid is a box and gets one, which is
  /// what shows where an imported phantom actually sits.
  template <typename real_t>
  void AddVolume(const geom::Volume<real_t>& v, unsigned int c) {
    if (v.solid.type == geom::SolidType::kBox || v.solid.type == geom::SolidType::kVoxelGrid) {
      AddBox(v.xform, v.solid.p[0], v.solid.p[1], v.solid.p[2], c);
    }
  }

  void clear() {
    x0.clear(); y0.clear(); z0.clear(); x1.clear(); y1.clear(); z1.clear(); rgb.clear();
  }
  std::size_t Size() const { return rgb.size(); }
};

}  // namespace g4gpu::vis
