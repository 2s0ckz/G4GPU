// A float copy of the scene, for the render pass only.
//
// WHY THIS EXISTS
//
// The renderer calls the transport's geometry functions, which is deliberate - the picture
// then cannot disagree with the physics about where a surface is. But the transport is double,
// and on a GeForce card double is not a precision choice, it is a throughput cliff: one FP64
// unit per 64 FP32 units on sm_86. Measured on this card, one ray per pixel at 1440x880,
// `mesh_nearest_hit` alone costs
//
//     triangles      float      double
//        20,000    4.70 ms    89.57 ms
//       200,000   11.30 ms    88.94 ms
//     1,000,000   18.76 ms   103.75 ms
//
// and the shape of the double column is the point: 89.6, 88.9, 103.8 is FLAT against triangle
// count, because the pass is arithmetic-bound and the tree barely matters. In float the cost
// tracks the tree, which is the healthy regime. Measured end to end in the viewer, alternating
// two builds within one session, the whole render pass comes out 1.9 to 2.2x faster.
//
// The transport stays double, and that is not caution: float mesh geometry puts about 1e-5 mm
// of error into where a track stops, and the dose is the number this project is judged on. A
// picture has no such requirement - a pixel is a third of a millimetre at any useful zoom.
//
// IT IS ALL A CAST, AND HERE IS WHY THAT IS SAFE
//
// The worry worth having is the BVH. A box converted INWARD by one ulp no longer contains its
// own triangles, and the traversal rejects any subtree whose box the ray misses - so the mesh
// would not be shaded a little wrong, it would have a hole along the seam.
//
// It cannot happen, and not by luck. `geom::build_bvh` sets every box bound to the min or max
// of vertex COORDINATES - `lo[k] = min(lo[k], c)` over the vertices themselves - so each bound
// IS one of the numbers that will be compared against it. Rounding to float is monotonic, so
// `float(bound) <= float(vertex)` wherever `bound <= vertex`, and containment survives the
// cast exactly. The same argument covers a mesh solid's own p[0..5], which come from the same
// extremes.
//
// This was written the other way first: sixty lines recomputing every node's box from the
// float triangles in a post-order walk, on the strength of the worry rather than of the
// argument. The test meant to justify those lines asked whether a plain cast breaks
// containment, and the answer was no - in every one of 4095 nodes. So they went.
// `tests/test_float_render.cu` now asserts the property the cast relies on instead: if
// `build_bvh` ever COMPUTES a bound rather than picking one, that test fails and this comment
// is what it points at.
//
// The voxel arrays are shared rather than copied: cells are shorts and class layers are ints,
// so they mean the same thing in both precisions. For a 512^3 phantom that is 268 MB not
// copied.
#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "geometry/mesh.cuh"
#include "geometry/navigator.cuh"

namespace g4gpu::vis {

/// The host-side pools a float copy is built from.
///
/// Spans rather than the scene type, so this header depends on `geometry/` alone and both the
/// builder and the viewer can fill it from whatever they hold.
struct HostGeometry {
  const geom::Volume<double>* volumes = nullptr;
  int n_volumes = 0;
  int world = 0;
  const geom::Solid<double>* solids = nullptr;
  int n_solids = 0;
  const geom::Transform<double>* xforms = nullptr;
  int n_xforms = 0;
  const double* aux = nullptr;
  int n_aux = 0;
  const double* tri = nullptr;
  int n_tri = 0;   ///< reals, not triangles: nine per triangle
  const double* bvh = nullptr;
  int n_bvh = 0;   ///< reals, not nodes: kBvhStride per node
};

/// The host-side result of converting a scene to float. Everything but the voxel arrays,
/// which are shared.
struct FloatPools {
  std::vector<geom::Volume<float>> volumes;
  std::vector<geom::Solid<float>> solids;
  std::vector<geom::Transform<float>> xforms;
  std::vector<float> aux, tri, bvh;
};

namespace detail {

inline geom::Solid<float> ToFloat(const geom::Solid<double>& s) {
  geom::Solid<float> o{};
  o.type = s.type;
  for (int k = 0; k < 8; ++k) { o.p[k] = static_cast<float>(s.p[k]); }
  o.xform = s.xform;
  o.a = s.a;
  o.b = s.b;
  return o;
}

inline geom::Transform<float> ToFloat(const geom::Transform<double>& x) {
  geom::Transform<float> o{};
  for (int k = 0; k < 9; ++k) { o.rot[k] = static_cast<float>(x.rot[k]); }
  o.trans = Vec3<float>{static_cast<float>(x.trans.x), static_cast<float>(x.trans.y),
                        static_cast<float>(x.trans.z)};
  o.identity = x.identity;
  return o;
}

}  // namespace detail

/// Converts a scene to float, on the host.
///
/// A free function over host vectors so that what it produced can be checked without a device
/// - see tests/test_float_render.cu, which walks the tree it hands back.
inline void ConvertToFloat(const HostGeometry& h, FloatPools& out) {
  out = FloatPools{};
  out.tri.resize(static_cast<std::size_t>(h.n_tri));
  for (int i = 0; i < h.n_tri; ++i) {
    out.tri[static_cast<std::size_t>(i)] = static_cast<float>(h.tri[i]);
  }
  // The node indices - `first` and `count` - ride in the same array as the box, as reals. A
  // float holds an integer exactly up to 2^24, so a mesh of more than 16.7 million triangles
  // would start losing them. That is also more triangles than this renderer is useful at, and
  // the point is only that the limit is known rather than assumed away.
  out.bvh.resize(static_cast<std::size_t>(h.n_bvh));
  for (int i = 0; i < h.n_bvh; ++i) {
    out.bvh[static_cast<std::size_t>(i)] = static_cast<float>(h.bvh[i]);
  }
  out.solids.resize(static_cast<std::size_t>(h.n_solids));
  for (int i = 0; i < h.n_solids; ++i) {
    out.solids[static_cast<std::size_t>(i)] = detail::ToFloat(h.solids[i]);
  }
  out.volumes.resize(static_cast<std::size_t>(h.n_volumes));
  for (int i = 0; i < h.n_volumes; ++i) {
    const geom::Volume<double>& v = h.volumes[i];
    geom::Volume<float>& o = out.volumes[static_cast<std::size_t>(i)];
    o.solid = detail::ToFloat(v.solid);
    o.xform = detail::ToFloat(v.xform);
    o.layer = v.layer;
    o.material = v.material;
    o.score_index = v.score_index;
    o.score_per_voxel = v.score_per_voxel;
    o.has_class_layers = v.has_class_layers;
    o.layer_lo = v.layer_lo;
    o.layer_hi = v.layer_hi;
  }
  out.xforms.resize(static_cast<std::size_t>(h.n_xforms));
  for (int i = 0; i < h.n_xforms; ++i) {
    out.xforms[static_cast<std::size_t>(i)] = detail::ToFloat(h.xforms[i]);
  }
  out.aux.resize(static_cast<std::size_t>(h.n_aux));
  for (int i = 0; i < h.n_aux; ++i) {
    out.aux[static_cast<std::size_t>(i)] = static_cast<float>(h.aux[i]);
  }
}

/// A float `geom::Geometry` on the device, rebuilt whenever the scene is.
class FloatGeometry {
 public:
  FloatGeometry() = default;
  FloatGeometry(const FloatGeometry&) = delete;
  FloatGeometry& operator=(const FloatGeometry&) = delete;
  ~FloatGeometry() { Free(); }

  /// True once Build has produced something worth rendering.
  bool ready() const { return ready_; }
  const geom::Geometry<float>& geometry() const { return g_; }

  /// @param voxels the voxel arrays as already uploaded for the transport. Shorts and ints,
  ///        so the two precisions share them; passing the double geometry's own is correct.
  void Build(const HostGeometry& h, const geom::VoxelStore<float>& voxels) {
    Free();
    if (h.n_volumes <= 0) { return; }
    FloatPools p;
    ConvertToFloat(h, p);
    d_volumes_ = Upload(p.volumes);
    d_solids_ = Upload(p.solids);
    d_xforms_ = Upload(p.xforms);
    d_aux_ = Upload(p.aux);
    d_tri_ = Upload(p.tri);
    d_bvh_ = Upload(p.bvh);

    g_ = geom::Geometry<float>{};
    g_.volumes = d_volumes_;
    g_.n_volumes = h.n_volumes;
    g_.world = h.world;
    g_.store.solids = d_solids_;
    g_.store.xforms = d_xforms_;
    g_.store.aux = d_aux_;
    g_.store.tri = d_tri_;
    g_.store.bvh = d_bvh_;
    g_.voxels = voxels;
    ready_ = true;
  }

  void Free() {
    cudaFree(const_cast<geom::Volume<float>*>(d_volumes_));
    cudaFree(const_cast<geom::Solid<float>*>(d_solids_));
    cudaFree(const_cast<geom::Transform<float>*>(d_xforms_));
    cudaFree(const_cast<float*>(d_aux_));
    cudaFree(const_cast<float*>(d_tri_));
    cudaFree(const_cast<float*>(d_bvh_));
    d_volumes_ = nullptr;
    d_solids_ = nullptr;
    d_xforms_ = nullptr;
    d_aux_ = nullptr;
    d_tri_ = nullptr;
    d_bvh_ = nullptr;
    g_ = geom::Geometry<float>{};
    bytes_ = 0;
    ready_ = false;
  }

  /// Bytes on the device. A phantom's cells are not among them - those are shared with the
  /// transport, which is the whole reason the voxel arrays are not converted.
  std::size_t bytes() const { return bytes_; }

 private:
  template <typename T>
  T* Upload(const std::vector<T>& v) {
    if (v.empty()) { return nullptr; }
    T* d = nullptr;
    const std::size_t n = sizeof(T) * v.size();
    if (cudaMalloc(&d, n) != cudaSuccess) {
      std::printf("FATAL: cannot allocate %zu bytes for the float render geometry\n", n);
      std::exit(2);
    }
    if (cudaMemcpy(d, v.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("FATAL: cannot upload the float render geometry\n");
      std::exit(2);
    }
    bytes_ += n;
    return d;
  }

  geom::Geometry<float> g_{};
  const geom::Volume<float>* d_volumes_ = nullptr;
  const geom::Solid<float>* d_solids_ = nullptr;
  const geom::Transform<float>* d_xforms_ = nullptr;
  const float* d_aux_ = nullptr;
  const float* d_tri_ = nullptr;
  const float* d_bvh_ = nullptr;
  std::size_t bytes_ = 0;
  bool ready_ = false;
};

}  // namespace g4gpu::vis
