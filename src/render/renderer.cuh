// GPU renderer for the B1 scene.
//
// The geometry pass is a ray cast that reuses geom::dist_in and geom::normal_at - the very
// same functions transport uses for distance-to-boundary - so visualization adds no new
// geometry code and cannot disagree with the physics about where a surface is.
//
// Depth resolution uses a packed 64-bit framebuffer: high 32 bits are the IEEE bit pattern
// of the depth (monotonic for positive floats), low 32 bits are packed RGB. A single
// atomicMin then does a correct nearest-wins depth test, which lets the wireframe and
// trajectory passes composite against the ray-cast surfaces with no separate z-buffer.
#pragma once
#include <cmath>
#include "core/vec3.cuh"
#include "geometry/navigator.cuh"
#include "geometry/safety.cuh"
#include "render/trajectory.cuh"

namespace g4gpu::vis {

using Vec3f = Vec3<float>;

// ---------------------------------------------------------------- packed framebuffer

/// Colour and COVERAGE in the framebuffer's low word: 0xAARRGGBB.
///
/// The alpha byte was spare, and its absence is why a nearly transparent volume looked black.
/// The geometry pass accumulates PREMULTIPLIED colour - each surface contributes
/// `alpha * (1 - alpha_so_far) * colour` - so a volume at 3% opacity writes about 3% of its
/// colour and nothing else. Written straight out that is a very dark pixel; what it MEANS is
/// "3% of this colour and 97% of whatever is behind", and behind it is the viewer's gradient.
/// So the coverage travels with the colour and the resolve pass finishes the compositing.
__host__ __device__ inline unsigned int pack_rgba(int r, int g, int b, int a) {
  return (static_cast<unsigned int>(a & 0xFF) << 24) |
         (static_cast<unsigned int>(r & 0xFF) << 16) |
         (static_cast<unsigned int>(g & 0xFF) << 8) | static_cast<unsigned int>(b & 0xFF);
}

/// Opaque, which is what every caller but the geometry pass means: a wireframe edge and a
/// trajectory segment are lines, not glass.
__host__ __device__ inline unsigned int pack_rgb(int r, int g, int b) {
  return pack_rgba(r, g, b, 255);
}

__device__ inline unsigned long long pack_pixel(float depth, unsigned int rgb) {
  // Positive floats compare correctly as unsigned ints, so depth ordering is preserved.
  const unsigned int d = __float_as_uint(depth);
  return (static_cast<unsigned long long>(d) << 32) | rgb;
}

constexpr unsigned long long kEmptyPixel = 0xFFFFFFFFFFFFFFFFull;

// ---------------------------------------------------------------- camera

struct Camera {
  Vec3f eye;
  Vec3f forward;  ///< unit
  Vec3f right;    ///< unit
  Vec3f up;       ///< unit
  float tan_half_fov;
  float aspect;
  int width;
  int height;

  __host__ __device__ Vec3f ray_dir(int px, int py) const {
    const float sx = (2.0f * (px + 0.5f) / width - 1.0f) * aspect * tan_half_fov;
    const float sy = (1.0f - 2.0f * (py + 0.5f) / height) * tan_half_fov;
    return normalize(forward + sx * right + sy * up);
  }

  /// World point to pixel coordinates plus view depth. Returns false if behind the camera.
  __host__ __device__ bool project(const Vec3f& p, float& px, float& py, float& depth) const {
    const Vec3f v = p - eye;
    depth = dot(v, forward);
    if (depth <= 1e-3f) { return false; }
    const float x = dot(v, right) / (depth * aspect * tan_half_fov);
    const float y = dot(v, up) / (depth * tan_half_fov);
    px = (x + 1.0f) * 0.5f * width;
    py = (1.0f - y) * 0.5f * height;
    return true;
  }
};

__host__ inline Camera make_camera(const Vec3f& eye, const Vec3f& target, const Vec3f& up_hint,
                                   float fov_deg, int width, int height) {
  Camera c{};
  c.eye = eye;
  c.forward = normalize(target - eye);
  c.right = normalize(cross(c.forward, up_hint));
  c.up = cross(c.right, c.forward);
  c.tan_half_fov = std::tan(fov_deg * 3.14159265f / 180.0f * 0.5f);
  c.aspect = static_cast<float>(width) / static_cast<float>(height);
  c.width = width;
  c.height = height;
  return c;
}

// ---------------------------------------------------------------- geometry pass

/// Per-volume rendering style. Solid volumes are ray cast; wireframe volumes are drawn as
/// edges in the line pass, which is how Geant4 shows a containing world by default.
struct VolumeStyle {
  bool solid = false;
  unsigned char r = 200, g = 200, b = 200;
  /// 255 is opaque. Anything less is composited front to back, which is what makes a nested
  /// detector legible: the outer shell at 0.3 and the target at 1.0 shows both.
  ///
  /// It defaults to opaque, and that default is load-bearing. `render_geometry` skips a volume
  /// with `a == 0`, so a caller that fills in r, g, b and `solid` and forgets this one gets a
  /// scene in which nothing renders at all - which is what happened to the viewer for as long
  /// as this field existed: it built its styles with `std::vector<VolumeStyle> styles(n)`,
  /// which zero-initialises, set every other field, and drew no solid geometry ever again. The
  /// wireframe pass kept working, so the picture looked like a deliberate wireframe view
  /// rather than a bug, and every automated check passed because they all count on the lines.
  ///
  /// Forgetting it now means opaque, which is the harmless direction to be wrong in.
  unsigned char a = 255;
};

/// Ray casts the solid volumes, front to back, with the layer rule and transparency.
///
/// One thread per pixel. The walk, rather than "nearest entry point wins", exists for two
/// reasons that turn out to be the same reason:
///
///   * **the layer rule.** A higher layer wins wherever two volumes overlap - that is what
///     replaces the mother/daughter tree - so the *transport* never sees the part of a lower
///     volume that a higher one covers. A renderer that drew the nearest surface drew that
///     part anyway, and the picture disagreed with the physics: an object half-buried in
///     another appeared whole. Ownership is asked of `locate()`, which is the same function
///     the stepper asks, so the two cannot drift.
///
///   * **transparency.** Compositing needs the surfaces in order, and once they are in order
///     the ownership test is free.
///
/// Capped at kMaxLayers surfaces per pixel. A ray through a detector crosses a handful; the
/// cap is what stops a pathological model from making one pixel cost a thousand.
template <typename real_t>
/// @param voxel_class per cell, parallel to geometry.voxels.material and indexed the same way
///        (VoxelGrid::index already carries the per-volume offset). Null renders voxel volumes
///        as the single box they were always rendered as.
/// @param class_rgba  0xAARRGGBB per class, concatenated over volumes. A voxel solid's
///        p[6] says where its own run starts and p[7] how many classes long it is.
__global__ void render_geometry(geom::Geometry<real_t> geometry, const VolumeStyle* styles,
                                Camera cam, unsigned long long* fb,
                                bool grid_lines = true,
                                const short* voxel_class = nullptr,
                                const unsigned int* class_rgba = nullptr) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= cam.width || py >= cam.height) { return; }

  const Vec3f d3 = cam.ray_dir(px, py);
  const Vec3<real_t> origin{real_t(cam.eye.x), real_t(cam.eye.y), real_t(cam.eye.z)};
  const Vec3<real_t> dir{real_t(d3.x), real_t(d3.y), real_t(d3.z)};

  constexpr int kMaxLayers = 8;
  // A nudge past each entry point, so the ownership test and the next search start inside the
  // surface just crossed rather than exactly on it.
  //
  // TEN TIMES LARGER IN FLOAT, and not by taste. A coordinate of 500 mm - this project's
  // default world is 500 - has a float ulp of 6e-5 mm, so 1e-4 is under two ulps: adding it
  // may not move the value at all, and a nudge that does not move the point leaves the search
  // standing on the surface it just crossed, finding it again until the layer cap. 1e-3 is
  // sixteen ulps there and still a thirtieth of a pixel at any useful zoom. The transport uses
  // the same two numbers for the same reason - see kPushDistance.
  const real_t kNudge = (sizeof(real_t) == 4) ? real_t(1e-3) : real_t(1e-4);

  float acc_r = 0, acc_g = 0, acc_b = 0;
  float acc_a = 0;
  float first_t = -1;
  real_t travelled = 0;

  // Is this volume a voxel grid that will be drawn cell by cell? Asked in two places - the
  // surface search and the walk itself - and they must agree, or a grid is re-entered by a
  // rule that then declines to draw it and the pixel loops until the layer cap.
  auto per_cell_grid = [&](int v) {
    return geometry.volumes[v].solid.type == geom::SolidType::kVoxelGrid
           && voxel_class != nullptr && class_rgba != nullptr
           && static_cast<int>(geometry.volumes[v].solid.p[7]) > 0;
  };
  // The nearest entry ahead of @p p, for the volumes that OUTRANK a grid: it is where the
  // grid stops owning the ray.
  //
  // Plain dist_in, and no shortcut for a mesh even though the fast path above has one. What a
  // shortcut here would buy is a cheaper ownership scan for a scene holding both a per-cell
  // voxel grid and a mesh above it; what it costs is a third inlined copy of the BVH walk in a
  // kernel that already spills. The ownership work in this function is done once per
  // composited layer, not per triangle, so the argument that made the fast path worth 4x does
  // not apply to it. Simplicity wins by default when neither side is measured.
  //
  // The register situation, since it is easy to check and easy to get wrong: these additions
  // took render_geometry from 250 registers and no spilling to 255 and 222 bytes of spill
  // stores (nvcc -Xptxas -v). That is real and it costs nothing measurable - paired A/B on
  // two binaries differing only in this file, t = 0.43 on 4 dof. Removing this shortcut did
  // not reduce the spilling either, and __noinline__ on the ownership helpers made it worse,
  // because an ABI call forces the caller to save its live registers. See docs/RISK.md V22.
  // TWO FACES IN THE SAME PLACE: THE ONE THAT OWNS THE SPACE IS THE ONE TO DRAW.
  //
  // The search used to take the first volume found at the nearest distance, and the ownership
  // test below then threw it away if a higher layer owned the space behind it - which for two
  // coincident faces is exactly what happens. The next iteration starts just inside both, so
  // both are "already inside" and neither is entered again: the pixel drew nothing at all.
  // Reported as two perfectly overlapping faces rendering as a hole, even on different layers.
  //
  // So among surfaces within a nudge of each other - the distance at which this loop already
  // treats two surfaces as one - the higher-ranked volume wins. The rank used is the volume's
  // highest possible one, because the hit point is not known yet; `locate` below still has the
  // last word, so a grid whose class loses at that point behaves exactly as it did.
  auto better = [&](int v, real_t tv, int best_vol, real_t best_t) {
    if (best_vol < 0 || tv < best_t - kNudge) { return true; }
    if (tv > best_t + kNudge) { return false; }
    return geom::volume_rank(geom::layer_hi_of(geometry, v), v)
           > geom::volume_rank(geom::layer_hi_of(geometry, best_vol), best_vol);
  };
  auto entry_ahead = [&](int v, const Vec3<real_t>& p) {
    const auto& u = geometry.volumes[v];
    return geom::dist_in(geometry.store, u.solid, geom::to_local(u.xform, p),
                         geom::dir_to_local(u.xform, dir));
  };

  for (int layer = 0; layer < kMaxLayers && acc_a < 0.995f; ++layer) {
    const Vec3<real_t> from = origin + travelled * dir;
    real_t best_t = geom::kInfinity<real_t>();
    int best_vol = -1;
    int best_tri = -1;   // for a mesh: the triangle that was hit, so its face normal is free
    for (int v = 0; v < geometry.n_volumes; ++v) {
      if (!styles[v].solid || styles[v].a == 0) { continue; }
      const auto& vol = geometry.volumes[v];

      // A MESH TAKES THE CHEAP PATH: one BVH walk, and the normal comes off the triangle.
      //
      // The generic path below is the transport's, and it is right for the transport and
      // ruinous for a picture. Per pixel, per layer, a mesh volume was paying:
      //
      //   inside_volume  -> mesh_inside, up to 4 full parity counts
      //   dist_in        -> mesh_dist: one nearest hit plus up to 16 more parity counts,
      //                     verifying that the hit is not a graze
      //   normal_at      -> numerical_normal: six containment tests, up to 24 parity counts
      //
      // A parity count cannot be culled the way a nearest-hit walk can - there is no
      // best-so-far to reject a subtree against - so it visits every leaf along the ray. That
      // is what made a 100k-triangle import unusable while the BVH itself was fine.
      //
      // A picture needs none of it. The nearest hit ahead is the surface to draw, a graze is
      // harmless to shade, and the face normal of the winning triangle is EXACT where the
      // numerical gradient was an approximation. The inside test goes too: with a
      // strictly-ahead hit the search cannot re-find the surface it just crossed, which is
      // what that test was there to prevent.
      if (vol.solid.type == geom::SolidType::kMesh) {
        real_t edge = 0;
        int tri = -1;
        const real_t t = geom::mesh_nearest_hit(
            geometry.store, vol.solid, geom::to_local(vol.xform, from),
            geom::dir_to_local(vol.xform, dir), geom::kSurfTolerance<real_t>(),
            geom::kInfinity<real_t>(), edge, &tri);
        if (t < geom::kInfinity<real_t>() && better(v, t, best_vol, best_t)) {
          if (t < best_t) { best_t = t; }
          best_vol = v;
          best_tri = tri;
        }
        continue;
      }

      // A volume the ray is already inside has no *entry* surface ahead of it, and asking
      // dist_in from inside gives a distance of about zero - so without this test the search
      // re-finds the volume just entered, paints its front face again, and does that until
      // the accumulated alpha saturates. The effect is that a half-transparent box renders
      // fully opaque and nothing behind it is ever reached. That is the whole bug.
      if (geom::inside_volume(geometry, v, from)) {
        // A GRID DRAWN CELL BY CELL IS THE EXCEPTION, because its interior is not one
        // surface. The walk below stops where a higher layer takes the space over, so the
        // cells BEYOND whatever covers the grid are still to be drawn - and the ray is inside
        // the grid the whole time. Without this they were simply lost.
        if (!per_cell_grid(v)) { continue; }
        // Who owns this point: the same question locate answers for the transport. If it is
        // the grid, the grid has already been walked from here and there is nothing ahead of
        // it; if it is something else, that something outranks the grid (locate returns the
        // highest layer containing the point) and the grid resumes where it ends.
        const int own = geom::locate(geometry, from);
        if (own < 0 || own == v) { continue; }
        const auto& cov = geometry.volumes[own];
        const real_t t = geom::dist_out(geometry.store, cov.solid,
                                        geom::to_local(cov.xform, from),
                                        geom::dir_to_local(cov.xform, dir));
        if (t < geom::kInfinity<real_t>() && better(v, t, best_vol, best_t)) {
          if (t < best_t) { best_t = t; }
          best_vol = v;
          best_tri = -1;
        }
        continue;
      }
      // THE RAY STARTS NEAR THE SOLID, NOT AT THE CAMERA.
      //
      // The generic engine solves a quadratic in the ray parameter, and its discriminant is
      // b^2 - 4ac. For a 60 mm sphere seen from 1500 mm that is 9.00e6 - 8.99e6 = 1.44e4: a
      // difference of two numbers that agree to three digits, which in float leaves four. The
      // root comes out about 0.004 mm wrong, `is_crossing_to` probes 0.001 mm either side of
      // it, both probes land on the same side of the surface, no crossing is seen, and the
      // pixel draws nothing. Measured: an orb keeps 100% of its rays at 200 mm, 47% at 1500
      // and 10% at 4000 - which is what "the sphere renders with speckle" was.
      //
      // Moving the origin to the solid's bounding sphere makes b and c both O(radius), so
      // nothing cancels and the hit rate stops depending on where the camera is. The skip is
      // sound because no surface of the solid lies before the bounding sphere; where no bound
      // is known - a box, a boolean, a grid - bounding_radius returns 0, and those shapes have
      // closed forms or flat faces and were never affected.
      //
      // A picture's arithmetic, not the transport's: the transport is double, where the same
      // discriminant keeps twelve digits.
      const Vec3<real_t> ql = geom::to_local(vol.xform, from);
      const Vec3<real_t> dl = geom::dir_to_local(vol.xform, dir);
      const real_t reach = geom::bounding_radius(vol.solid);
      real_t skip = real_t(0);
      if (reach > real_t(0)) {
        // The closest the ray comes to the solid's own origin, less the bound. Never negative:
        // a ray already inside the bounding sphere starts where it is.
        const real_t approach = -dot(ql, dl);
        skip = approach - reach * real_t(1.01) - real_t(1);
        if (skip < real_t(0)) { skip = real_t(0); }
      }
      // geometry.store, not the store-free overload. That overload passes an empty SolidStore,
      // whose `solids`, `aux`, `tri` and `bvh` are all null - so a boolean operand lookup
      // dereferences null and a mesh finds no triangles.
      const real_t t = geom::dist_in(geometry.store, vol.solid, ql + skip * dl, dl);
      if (t < geom::kInfinity<real_t>() && better(v, skip + t, best_vol, best_t)) {
        if (skip + t < best_t) { best_t = skip + t; }
        best_vol = v;
        best_tri = -1;
      }
    }
    if (best_vol < 0 || best_t >= geom::kInfinity<real_t>()) { break; }

    const auto& vol = geometry.volumes[best_vol];
    const Vec3<real_t> hit = from + best_t * dir;
    const real_t t_hit = travelled + best_t;   ///< distance from the eye to this surface
    travelled += best_t + kNudge;

    // Is this surface actually the boundary of the volume that *owns* the space behind it?
    // If a higher layer covers this region, the answer is no, and the surface is not there as
    // far as the transport is concerned - so it is not drawn either. The search then carries
    // on from just inside, which finds the covering volume's own surface next.
    //
    // NOT `locate(...) == best_vol`, though that is the same answer. The walk has just crossed
    // into this volume, so its containment is already known - and asking a mesh again is a
    // mesh_inside parity count, the one traversal the nearest-hit walk exists to avoid. It was
    // being paid once per composited layer, which an opaque volume hides (the walk stops at
    // its first surface) and a translucent one does not. See geom::owns_contained_point.
    // No special case for an absent entering cell, and there was one while absence was a
    // layer. An absent cell now reports the volume's own layer here - the layer array holds
    // that for a class the scene does not contain, since the number is never read as a
    // priority - so the question asked at the grid's surface is the one it was before the
    // feature existed, and the march below skips the cells that are not there.
    if (!geom::owns_contained_point(geometry, best_vol, hit + kNudge * dir)) { continue; }

    const Vec3<real_t> local_dir = geom::dir_to_local(vol.xform, dir);
    const Vec3<real_t> hit_local = geom::to_local(vol.xform, hit);
    // The mesh's normal was found by the walk that found the hit; everything else asks.
    Vec3<real_t> n =
        (best_tri >= 0)
            ? geom::dir_to_global(vol.xform,
                                  geom::mesh_triangle_normal<real_t>(geometry.store, best_tri))
            : geom::dir_to_global(vol.xform,
                                  geom::normal_at(geometry.store, vol.solid, hit_local));
    // Two-sided lighting: a head-on light plus a little ambient, so curvature reads.
    real_t ndl = -dot(n, dir);
    if (ndl < real_t(0)) { ndl = -ndl; }
    const float shade = 0.25f + 0.75f * static_cast<float>(ndl);

    // A voxel volume whose cells are coloured individually is not one surface.
    //
    // Everything above found where the ray enters the grid's bounding box. For an ordinary
    // solid that is the surface to shade and there is nothing more to say; for a segmented
    // phantom it is the outside of a box with the anatomy inside it, and shading it is how an
    // imported CT came to draw as a featureless slab.
    //
    // So walk the cells. Each one contributes its class's colour and opacity, front to back,
    // by the same accumulation the outer loop uses for whole volumes - and a class at zero
    // opacity contributes nothing, which is what makes index 0 (air, background, outside the
    // patient) get out of the way and the tissue behind it visible.
    //
    // Shading is by the face the cell was entered through: VoxelWalk reports the axis, and the
    // normal is the unit vector along it. Without that every cell shades identically and the
    // result is fog with an outline. With it, the boundary between two classes reads as a
    // surface, which is what makes the picture anatomy rather than a colour field.
    if (per_cell_grid(best_vol)) {
      const geom::VoxelGrid<real_t> grid = geom::voxel_grid_of(vol.solid);
      // HOW FAR THE GRID OWNS THE RAY.
      //
      // A higher layer takes the space wherever it overlaps, so the cells under it are not
      // there as far as the transport is concerned and must not be drawn - the same rule the
      // `locate` test above applies to an ordinary surface. The walk had no such test: it
      // marched the grid's whole depth in one go and composited every cell along it,
      // including the ones sitting inside an opaque volume placed over the phantom. Front to
      // back, those cells arrive BEFORE that volume's own surface, so they were blended over
      // the top of it - which is a phantom showing through a solid object, reported exactly
      // that way.
      //
      // Style is deliberately not consulted: a volume that is hidden or wireframe still owns
      // its space, which is what `locate` says and therefore what the transport sees. Hiding
      // a box over a phantom leaves the hole it occupies, the same as it already does over an
      // ordinary solid.
      // WHICH IS A QUESTION PER CELL, once classes can be on different layers.
      //
      // A grid whose bone class outranks the box over it and whose air class does not has two
      // answers along one ray, so a single clamp cannot express it. What is constant per ray
      // is each candidate volume's ENTRY - so the scan happens once and records, for each
      // volume that could outrank any class here, how far away it starts and what it ranks.
      // The march then takes the nearest of those that outrank the cell it is looking at,
      // which is a handful of comparisons and no geometry calls.
      //
      // For a grid whose classes are all on one layer - every phantom until now - every cell
      // has the same rank and this collapses to exactly the single clamp it replaces.
      constexpr int kMaxCovers = 4;
      real_t cov_t[kMaxCovers];
      long long cov_rank[kMaxCovers];
      int n_cov = 0;
      {
        // The lowest rank any cell here can have. A volume that cannot beat this cannot cover
        // any cell of the grid, and for a uniform grid that is the whole test. Absent classes
        // contribute no layer to this minimum, because they contribute no layer at all -
        // G4Flatten computes the range over the classes that are present.
        const long long lo_rank = geom::volume_rank(geom::layer_lo_of(geometry, best_vol),
                                                    best_vol);
        for (int v = 0; v < geometry.n_volumes; ++v) {
          if (v == best_vol || !geom::could_outrank(geometry, v, lo_rank)) { continue; }
          const real_t t = entry_ahead(v, hit);
          if (t >= geom::kInfinity<real_t>()) { continue; }
          const long long r = geom::volume_rank(geom::layer_hi_of(geometry, v), v);
          if (n_cov < kMaxCovers) {
            cov_t[n_cov] = t;
            cov_rank[n_cov] = r;
            ++n_cov;
            continue;
          }
          // Full: keep the nearest, since a farther surface can only clamp later. Dropping
          // the farthest can only ever draw a cell that should have been covered, and only
          // in a scene with five higher-layer volumes over one phantom.
          int worst = 0;
          for (int k = 1; k < kMaxCovers; ++k) {
            if (cov_t[k] > cov_t[worst]) { worst = k; }
          }
          if (t < cov_t[worst]) {
            cov_t[worst] = t;
            cov_rank[worst] = r;
          }
        }
      }
      // p[6] and p[7] are spare on a voxel grid - p[0..2] is the half extent and
      // p[3..5] the cell counts - so the class run rides on the SOLID, exactly as the
      // cell-array offset rides in `a`. Nothing records which model solid a flattened
      // volume came from (the scene is built from the Geant4 placement tree), so a
      // per-volume style could not have carried it.
      const int cbase = static_cast<int>(vol.solid.p[6]);
      const int ccount = static_cast<int>(vol.solid.p[7]);
      geom::VoxelWalk<real_t> walk;
      // A nudge inside, so the entry point is unambiguously in the first cell rather than on
      // its face - the same reason the outer loop nudges past each surface it crosses, and the
      // same constant, because the comparison below adds one to the other.
      const Vec3<real_t> start = hit_local + kNudge * local_dir;
      bool covered = false;    ///< the walk ended because a higher layer took the space
      real_t end_t = 0;        ///< where it ended, measured from `hit`
      if (walk.Start(grid, start, local_dir)) {
        Vec3<real_t> face_n = n;   // the grid's own surface, for the first cell
        bool more = true;
        while (more && acc_a < 0.995f) {
          // This cell's own rank, and the nearest surface that outranks it. Both are the
          // volume's own when the grid has no per-class layers, and then t_own is the same
          // number for every cell.
          const int cell_layer =
              geom::voxel_cell_layer(geometry.voxels, grid, walk.Index(grid));

          // A CELL THAT IS NOT THERE IS SKIPPED, AND THE MARCH GOES ON.
          //
          // Skipped, not "covered". Both leave it unpainted and only one of them is right:
          // `covered` BREAKS the march - it means "a higher layer takes the space from here
          // on, hand the ray back to the outer search" - so ending the walk at the first
          // absent cell takes every cell behind it too. A phantom whose front class was nulled
          // disappeared entirely, far side included. That is what absence-as-a-low-layer did
          // all by itself, with no line of code saying so: the cell ranked below everything,
          // the first candidate cover outranked it, and the clamp fired. Asked as "is this
          // cell there" instead, it cannot happen.
          if (geom::voxel_cell_absent(geometry.voxels, grid, walk.Index(grid))) {
            more = walk.Next(grid);
            if (more) {
              Vec3<real_t> ln{real_t(0), real_t(0), real_t(0)};
              const real_t sgn = (walk.step[walk.axis] > 0) ? real_t(-1) : real_t(1);
              if (walk.axis == 0) { ln.x = sgn; }
              else if (walk.axis == 1) { ln.y = sgn; }
              else { ln.z = sgn; }
              face_n = geom::dir_to_global(vol.xform, ln);
            }
            continue;
          }

          const long long cell_rank = geom::volume_rank(
              (cell_layer == geom::kNoClassLayer) ? vol.layer : cell_layer, best_vol);
          real_t t_own = geom::kInfinity<real_t>();
          for (int k = 0; k < n_cov; ++k) {
            if (cov_rank[k] > cell_rank && cov_t[k] < t_own) { t_own = cov_t[k]; }
          }
          // walk.t is the entry of the cell about to be drawn, so a cell straddling the
          // boundary is drawn: the part of it the grid owns is in front of the surface.
          //
          // MEASURED FROM THE SAME PLACE, AND TIES GO TO THE COVER. walk.t counts from
          // `start`, which is kNudge past `hit`, while t_own counts from `hit` - so the raw
          // comparison is out by that nudge, and it is out in the direction of drawing a cell
          // the cover owns. A box whose face is FLUSH with a cell boundary is exactly where
          // that shows, and flush is not an unlikely arrangement: cells are on a lattice and
          // a box put over one lands on it. The cell was then composited in front of an
          // opaque surface at a quarter opacity, which tints it - one cell's worth of a
          // phantom showing through, which is the same bug in miniature.
          //
          // The surface tolerance decides the exact tie, in favour of the cover, so that
          // alignment gives a definite answer instead of one an ulp of rounding picks.
          if (walk.t + kNudge >= t_own - geom::kSurfTolerance<real_t>()) {
            covered = true;
            end_t = t_own;
            break;
          }
          const int idx = walk.Index(grid);
          // Null cells never reach here; they were skipped above, before the clamp.
          if (idx >= 0 && idx < geometry.voxels.count) {
            const int cls = static_cast<int>(voxel_class[idx]);
            if (cls >= 0 && cls < ccount) {
              const unsigned int rgba = class_rgba[cbase + cls];
              const float ca = static_cast<float>((rgba >> 24) & 0xFFu) * (1.0f / 255.0f);
              if (ca > 0.0f) {
                real_t nd = -dot(face_n, dir);
                if (nd < real_t(0)) { nd = -nd; }
                const float sh = 0.25f + 0.75f * static_cast<float>(nd);
                const float w = ca * (1.0f - acc_a);
                acc_r += w * static_cast<float>((rgba >> 16) & 0xFFu) * sh;
                acc_g += w * static_cast<float>((rgba >> 8) & 0xFFu) * sh;
                acc_b += w * static_cast<float>(rgba & 0xFFu) * sh;
                acc_a += w;
                if (first_t < 0) { first_t = static_cast<float>(t_hit + walk.t); }
              }
            }
          }
          more = walk.Next(grid);
          if (more) {
            // The face just crossed, in the grid's frame, taken to global.
            Vec3<real_t> ln{real_t(0), real_t(0), real_t(0)};
            const real_t sgn = (walk.step[walk.axis] > 0) ? real_t(-1) : real_t(1);
            if (walk.axis == 0) { ln.x = sgn; }
            else if (walk.axis == 1) { ln.y = sgn; }
            else { ln.z = sgn; }
            face_n = geom::dir_to_global(vol.xform, ln);
          }
        }
        if (!covered) { end_t = walk.t; }   // the far side of the grid, where Next() stopped
      }
      // The grid is accounted for as far as it owns the ray, so the search resumes there
      // rather than shading its box as well.
      //
      // A NUDGE SHORT of a covering surface, not past it: a volume the ray is already inside
      // has no entry surface ahead of it, so restarting inside the thing that covers the
      // phantom is how that thing would never get painted. Past the grid's own far side when
      // nothing covered it, which is the same reasoning the other way round.
      //
      // At least a nudge either way. A covering surface flush with the grid's front face
      // gives end_t = 0, and a pixel whose distance does not advance is a pixel that spins
      // here until the layer cap.
      real_t resume = covered ? (end_t - kNudge) : (end_t + kNudge);
      if (resume < kNudge) { resume = kNudge; }
      travelled = t_hit + resume;
      continue;
    }

    // Voxel gridlines.
    //
    // A voxel volume is one box to the geometry - containment and entry are a box's - so
    // nothing about its surface says how it is divided, and a 64x64x1 detector and a
    // 512x512x120 CT drew as the same grey slab. The cell boundaries are drawn here, at the
    // point where the surface is already known, by darkening the shade near one.
    //
    // Only the two axes tangent to the face that was hit: the third is at a boundary
    // everywhere on that face by construction - it *is* the face - and including it would
    // paint the whole thing dark.
    //
    // The legibility rule matters as much as the drawing. A line is worth a pixel or two
    // whatever the zoom, so its width is set in pixels and converted to millimetres at the
    // hit distance; and once a cell is smaller than a few pixels the lines are more ink than
    // the cells they separate, so below that they are not drawn at all. Without that, a real
    // CT zoomed out is a solid black rectangle.
    float grid_shade = 1.0f;
    if (grid_lines && vol.solid.type == geom::SolidType::kVoxelGrid) {
      const auto& sol = vol.solid;
      const float mm_per_px =
          2.0f * static_cast<float>(travelled) * cam.tan_half_fov / float(cam.height);
      const float q[3] = {static_cast<float>(hit_local.x), static_cast<float>(hit_local.y),
                          static_cast<float>(hit_local.z)};
      // Which face: the coordinate closest to its own half extent.
      int face = 0;
      float worst = -1;
      for (int k = 0; k < 3; ++k) {
        const float h = static_cast<float>(sol.p[k]);
        const float f = (h > 0) ? fabsf(q[k]) / h : 0.0f;
        if (f > worst) {
          worst = f;
          face = k;
        }
      }
      for (int k = 0; k < 3; ++k) {
        if (k == face) { continue; }
        const float h = static_cast<float>(sol.p[k]);
        int n = static_cast<int>(sol.p[3 + k] + 0.5f);
        if (n < 1) { n = 1; }
        const float cell = 2.0f * h / float(n);
        if (cell <= 0.0f) { continue; }

        // A line is worth a pixel or so whatever the zoom, so the width is set in pixels and
        // converted to millimetres at the hit distance.
        float half_w = 0.7f * mm_per_px;

        // Fine grids are *faded*, not dropped.
        //
        // Dropping them below a few pixels per cell was the previous rule, and it meant a CT
        // seen from across the room drew as a featureless slab with no indication that it was
        // voxelised at all - the one thing the gridlines exist to show. But drawing them at
        // full strength when the cells are finer than the pixels paints every pixel dark: the
        // lines stop separating cells and become the object.
        //
        // So the strength scales with how much of a cell one line would cover. At ten pixels
        // per cell the line is a line; at one pixel per cell it is a wash that says "this is
        // divided more finely than you can see", which is true and is what the ink is for.
        // Nothing is ever dropped, and there is still a toggle for turning them off.
        float strength = 1.0f;
        const float coverage = 2.0f * half_w / cell;   // fraction of a cell one line covers
        if (coverage > 0.25f) {
          // Fade from full at quarter coverage to a tenth at total coverage.
          const float t = (coverage < 1.0f) ? (coverage - 0.25f) / 0.75f : 1.0f;
          strength = 1.0f - 0.9f * t;
          if (half_w > 0.25f * cell) { half_w = 0.25f * cell; }
        }

        // Distance to the nearest cell boundary along this axis.
        const float u = (q[k] + h) / cell;
        float frac = u - floorf(u);
        if (frac > 0.5f) { frac = 1.0f - frac; }
        if (frac * cell < half_w) {
          const float shade = 1.0f - 0.55f * strength;
          if (shade < grid_shade) { grid_shade = shade; }
        }
      }
    }

    const VolumeStyle s = styles[best_vol];
    const float alpha = static_cast<float>(s.a) * (1.0f / 255.0f);
    // Front-to-back accumulation: each surface contributes what the ones in front of it have
    // left transparent.
    const float w = alpha * (1.0f - acc_a);
    acc_r += w * static_cast<float>(s.r) * shade * grid_shade;
    acc_g += w * static_cast<float>(s.g) * shade * grid_shade;
    acc_b += w * static_cast<float>(s.b) * shade * grid_shade;
    acc_a += w;
    if (first_t < 0) { first_t = static_cast<float>(travelled - kNudge); }
  }

  unsigned long long value = kEmptyPixel;
  if (first_t >= 0 && acc_a > 0.002f) {
    // Depth is the *first* surface, so tracks and edges sort against what is in front rather
    // than against the last thing accumulated.
    const int ir = static_cast<int>(acc_r);
    const int ig = static_cast<int>(acc_g);
    const int ib = static_cast<int>(acc_b);
    // The coverage as well as the colour. See pack_rgba: what is accumulated is premultiplied,
    // so the resolve pass has to know how much of the pixel it covers to put the background
    // behind the rest of it.
    int ia = static_cast<int>(acc_a * 255.0f + 0.5f);
    if (ia > 255) { ia = 255; }
    value = pack_pixel(first_t, pack_rgba(ir, ig, ib, ia));
  }
  fb[py * cam.width + px] = value;
}

// ---------------------------------------------------------------- line pass

/// Draws one screen-space line with a depth test, thickness in pixels.
///
/// @param depth_scale 1.0 depth-tests normally against the ray-cast surfaces. A small value
///        (e.g. 1e-3) compresses all line depths below any surface depth, so lines show
///        through solids while still sorting correctly among themselves - an x-ray view,
///        which is what makes tracks *inside* a volume visible. Geant4's default viewer is
///        see-through for the same reason.
__device__ inline void draw_line(unsigned long long* fb, const Camera& cam, const Vec3f& a,
                                 const Vec3f& b, unsigned int rgb, int thickness,
                                 float depth_scale = 1.0f) {
  float ax, ay, az, bx, by, bz;
  // Both endpoints must be in front of the camera; segments are short, so skipping a
  // straddling segment loses at most one step of a trajectory.
  if (!cam.project(a, ax, ay, az)) { return; }
  if (!cam.project(b, bx, by, bz)) { return; }

  const float dx = bx - ax, dy = by - ay;
  const int steps = static_cast<int>(fmaxf(fabsf(dx), fabsf(dy))) + 1;
  if (steps > 4096) { return; }  // degenerate projection, skip rather than stall a warp

  for (int i = 0; i <= steps; ++i) {
    const float f = static_cast<float>(i) / static_cast<float>(steps);
    const float x = ax + dx * f;
    const float y = ay + dy * f;
    const float z = az + (bz - az) * f;
    // Bias toward the viewer so a line lying exactly on a surface still shows.
    const unsigned long long v = pack_pixel(z * 0.999f * depth_scale, rgb);
    for (int oy = -thickness; oy <= thickness; ++oy) {
      for (int ox = -thickness; ox <= thickness; ++ox) {
        const int ix = static_cast<int>(x) + ox;
        const int iy = static_cast<int>(y) + oy;
        if (ix < 0 || iy < 0 || ix >= cam.width || iy >= cam.height) { continue; }
        atomicMin(&fb[iy * cam.width + ix], v);
      }
    }
  }
}

/// The colours a picture is drawn with, other than the volumes' own.
///
/// Passed to the kernels rather than baked in, so the GUI's Visualisation attributes window
/// can change them. The defaults are what the viewer has always used.
/// Trajectory colours, by charge, as Geant4's default trajectory model draws them: negative
/// red, neutral green, positive blue. There is no fourth class - every particle has a charge -
/// which is why the `other` colour this used to carry is gone.
struct Palette {
  unsigned int neutral = 0x3CDC5Au;   ///< 0xRRGGBB - green
  unsigned int negative = 0xFF3C3Cu;  ///< red
  unsigned int positive = 0x5082FFu;  ///< blue
  /// The background gradient runs from `bg_top` at the top of the image to `bg_bottom`.
  unsigned int bg_top = 0x0C0E16u;
  unsigned int bg_bottom = 0x202638u;
};

/// Rasterizes captured trajectory segments, one thread per segment.
__global__ void render_trajectories(TrajectoryBuffer traj, int n, Camera cam,
                                    unsigned long long* fb, int thickness,
                                    float depth_scale = 1.0f, Palette pal = Palette{}) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const Vec3f a{traj.x0[i], traj.y0[i], traj.z0[i]};
  const Vec3f b{traj.x1[i], traj.y1[i], traj.z1[i]};
  unsigned int rgb;
  switch (traj.kind[i]) {
    case kKindNegative: rgb = pal.negative; break;
    case kKindPositive: rgb = pal.positive; break;
    default:            rgb = pal.neutral;  break;
  }
  draw_line(fb, cam, a, b, rgb, thickness, depth_scale);
}

/// Rasterizes an explicit edge list, used for the wireframe volumes.
__global__ void render_edges(const float* ex0, const float* ey0, const float* ez0,
                             const float* ex1, const float* ey1, const float* ez1,
                             const unsigned int* ergb, int n, Camera cam,
                             unsigned long long* fb, int thickness) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  draw_line(fb, cam, Vec3f{ex0[i], ey0[i], ez0[i]}, Vec3f{ex1[i], ey1[i], ez1[i]}, ergb[i],
            thickness);
}

// ---------------------------------------------------------------- resolve

/// Background colour for pixels no surface or line reached: a vertical gradient between the
/// palette's two ends, so the image does not read as flat black and the user can change it.
__host__ __device__ inline void background(int py, int height, const Palette& pal, int& r,
                                          int& g, int& b) {
  const float t = static_cast<float>(py) / static_cast<float>(height > 0 ? height : 1);
  const int r0 = (pal.bg_top >> 16) & 0xFF, g0 = (pal.bg_top >> 8) & 0xFF;
  const int b0 = pal.bg_top & 0xFF;
  const int r1 = (pal.bg_bottom >> 16) & 0xFF, g1 = (pal.bg_bottom >> 8) & 0xFF;
  const int b1 = pal.bg_bottom & 0xFF;
  r = static_cast<int>(r0 + (r1 - r0) * t);
  g = static_cast<int>(g0 + (g1 - g0) * t);
  b = static_cast<int>(b0 + (b1 - b0) * t);
}

/// Unpacks to top-down 24-bit RGB, the layout write_png_rgb expects.
/// One pixel of the framebuffer, finished: the accumulated colour over the background.
///
/// THE COLOUR IN THE FRAMEBUFFER IS PREMULTIPLIED AND MAY COVER ONLY PART OF THE PIXEL. The
/// geometry pass adds `alpha * (1 - alpha_so_far) * colour` per surface, so a volume at 3%
/// opacity leaves 3% of its colour there and nothing else. Written out as-is that is a nearly
/// black pixel, which is what "a very low opacity looks like a black background" was. What it
/// means is 3% of the colour and 97% of whatever is behind, and behind it is the viewer's
/// gradient - so the rest of the pixel is filled in here.
///
/// One function rather than the same arithmetic in three kernels, because the three had already
/// drifted: only one of them was fixed first time round, and the two used for offscreen export
/// would have written a different picture from the one on screen.
__host__ __device__ inline void resolve_pixel(unsigned long long v, int py, int height,
                                              const Palette& pal, int& r, int& g, int& b) {
  if (v == kEmptyPixel) {
    background(py, height, pal, r, g, b);
    return;
  }
  const unsigned int rgba = static_cast<unsigned int>(v & 0xFFFFFFFFull);
  r = (rgba >> 16) & 0xFF;
  g = (rgba >> 8) & 0xFF;
  b = rgba & 0xFF;
  const int a = static_cast<int>((rgba >> 24) & 0xFFu);
  if (a >= 255) { return; }
  int br, bg, bb;
  background(py, height, pal, br, bg, bb);
  r += br * (255 - a) / 255;
  g += bg * (255 - a) / 255;
  b += bb * (255 - a) / 255;
  if (r > 255) { r = 255; }
  if (g > 255) { g = 255; }
  if (b > 255) { b = 255; }
}

__global__ void resolve_to_rgb(const unsigned long long* fb, unsigned char* rgb_out, int width,
                               int height, Palette pal = Palette{}) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) { return; }

  int r, g, b;
  resolve_pixel(fb[py * width + px], py, height, pal, r, g, b);
  unsigned char* p = rgb_out + (static_cast<size_t>(py) * width + px) * 3;
  p[0] = static_cast<unsigned char>(r);
  p[1] = static_cast<unsigned char>(g);
  p[2] = static_cast<unsigned char>(b);
}

/// Unpacks to top-down 0xAABBGGRR, which is what the immediate-mode UI draws into and what
/// GL_RGBA/GL_UNSIGNED_BYTE uploads on a little-endian host. Having the render land in the
/// same layout as the UI is what lets the panels be composited on the CPU for the cost of one
/// pass over the pixels they cover.
__global__ void resolve_to_rgba(const unsigned long long* fb, unsigned int* out, int width,
                                int height, Palette pal = Palette{}) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) { return; }

  int r, g, b;
  resolve_pixel(fb[py * width + px], py, height, pal, r, g, b);
  out[static_cast<size_t>(py) * width + px] = 0xFF000000u
                                              | (static_cast<unsigned>(b) << 16)
                                              | (static_cast<unsigned>(g) << 8)
                                              | static_cast<unsigned>(r);
}

/// Unpacks the framebuffer to 24-bit BGR rows for a bottom-up BMP, on a vertical gradient
/// background so empty pixels are not flat black.
__global__ void resolve_to_bgr(const unsigned long long* fb, unsigned char* bgr, int width,
                               int height) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) { return; }

  int r, g, b;
  resolve_pixel(fb[py * width + px], py, height, Palette{}, r, g, b);
  // BMP rows run bottom-up.
  const int row = height - 1 - py;
  unsigned char* p = bgr + (static_cast<size_t>(row) * width + px) * 3;
  p[0] = static_cast<unsigned char>(b);
  p[1] = static_cast<unsigned char>(g);
  p[2] = static_cast<unsigned char>(r);
}

}  // namespace g4gpu::vis
