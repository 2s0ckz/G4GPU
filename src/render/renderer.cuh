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
#include "render/trajectory.cuh"

namespace g4gpu::vis {

using Vec3f = Vec3<float>;

// ---------------------------------------------------------------- packed framebuffer

__host__ __device__ inline unsigned int pack_rgb(int r, int g, int b) {
  return (static_cast<unsigned int>(r & 0xFF) << 16) |
         (static_cast<unsigned int>(g & 0xFF) << 8) | static_cast<unsigned int>(b & 0xFF);
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
__global__ void render_geometry(geom::Geometry<real_t> geometry, const VolumeStyle* styles,
                                Camera cam, unsigned long long* fb,
                                bool grid_lines = true) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= cam.width || py >= cam.height) { return; }

  const Vec3f d3 = cam.ray_dir(px, py);
  const Vec3<real_t> origin{real_t(cam.eye.x), real_t(cam.eye.y), real_t(cam.eye.z)};
  const Vec3<real_t> dir{real_t(d3.x), real_t(d3.y), real_t(d3.z)};

  constexpr int kMaxLayers = 8;
  // A nudge past each entry point, so the ownership test and the next search start inside the
  // surface just crossed rather than exactly on it.
  const real_t kNudge = real_t(1e-4);

  float acc_r = 0, acc_g = 0, acc_b = 0;
  float acc_a = 0;
  float first_t = -1;
  real_t travelled = 0;

  for (int layer = 0; layer < kMaxLayers && acc_a < 0.995f; ++layer) {
    const Vec3<real_t> from = origin + travelled * dir;
    real_t best_t = geom::kInfinity<real_t>();
    int best_vol = -1;
    for (int v = 0; v < geometry.n_volumes; ++v) {
      if (!styles[v].solid || styles[v].a == 0) { continue; }
      // A volume the ray is already inside has no *entry* surface ahead of it, and asking
      // dist_in from inside gives a distance of about zero - so without this test the search
      // re-finds the volume just entered, paints its front face again, and does that until
      // the accumulated alpha saturates. The effect is that a half-transparent box renders
      // fully opaque and nothing behind it is ever reached. That is the whole bug.
      if (geom::inside_volume(geometry, v, from)) { continue; }
      const auto& vol = geometry.volumes[v];
      // geometry.store, not the store-free overload. That overload passes an empty SolidStore,
      // whose `solids`, `aux`, `tri` and `bvh` are all null - so a boolean operand lookup
      // dereferences null and a mesh finds no triangles.
      const real_t t = geom::dist_in(geometry.store, vol.solid, geom::to_local(vol.xform, from),
                                     geom::dir_to_local(vol.xform, dir));
      if (t < best_t) {
        best_t = t;
        best_vol = v;
      }
    }
    if (best_vol < 0 || best_t >= geom::kInfinity<real_t>()) { break; }

    const auto& vol = geometry.volumes[best_vol];
    const Vec3<real_t> hit = from + best_t * dir;
    travelled += best_t + kNudge;

    // Is this surface actually the boundary of the volume that *owns* the space behind it?
    // If a higher layer covers this region, the answer is no, and the surface is not there as
    // far as the transport is concerned - so it is not drawn either. The search then carries
    // on from just inside, which finds the covering volume's own surface next.
    if (geom::locate(geometry, hit + kNudge * dir) != best_vol) { continue; }

    const Vec3<real_t> local_dir = geom::dir_to_local(vol.xform, dir);
    const Vec3<real_t> hit_local = geom::to_local(vol.xform, hit);
    Vec3<real_t> n =
        geom::dir_to_global(vol.xform, geom::normal_at(geometry.store, vol.solid, hit_local));
    // Two-sided lighting: a head-on light plus a little ambient, so curvature reads.
    real_t ndl = -dot(n, dir);
    if (ndl < real_t(0)) { ndl = -ndl; }
    const float shade = 0.25f + 0.75f * static_cast<float>(ndl);

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
    value = pack_pixel(first_t, pack_rgb(ir, ig, ib));
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
__device__ inline void background(int py, int height, const Palette& pal, int& r, int& g,
                                  int& b) {
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
__global__ void resolve_to_rgb(const unsigned long long* fb, unsigned char* rgb_out, int width,
                               int height, Palette pal = Palette{}) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) { return; }

  const unsigned long long v = fb[py * width + px];
  int r, g, b;
  if (v == kEmptyPixel) {
    background(py, height, pal, r, g, b);
  } else {
    const unsigned int rgb = static_cast<unsigned int>(v & 0xFFFFFFFFull);
    r = (rgb >> 16) & 0xFF;
    g = (rgb >> 8) & 0xFF;
    b = rgb & 0xFF;
  }
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

  const unsigned long long v = fb[py * width + px];
  int r, g, b;
  if (v == kEmptyPixel) {
    background(py, height, pal, r, g, b);
  } else {
    const unsigned int rgb = static_cast<unsigned int>(v & 0xFFFFFFFFull);
    r = (rgb >> 16) & 0xFF;
    g = (rgb >> 8) & 0xFF;
    b = rgb & 0xFF;
  }
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

  const unsigned long long v = fb[py * width + px];
  int r, g, b;
  if (v == kEmptyPixel) {
    const float t = static_cast<float>(py) / static_cast<float>(height);
    r = static_cast<int>(12 + 20 * t);
    g = static_cast<int>(14 + 24 * t);
    b = static_cast<int>(22 + 34 * t);
  } else {
    const unsigned int rgb = static_cast<unsigned int>(v & 0xFFFFFFFFull);
    r = (rgb >> 16) & 0xFF;
    g = (rgb >> 8) & 0xFF;
    b = rgb & 0xFF;
  }
  // BMP rows run bottom-up.
  const int row = height - 1 - py;
  unsigned char* p = bgr + (static_cast<size_t>(row) * width + px) * 3;
  p[0] = static_cast<unsigned char>(b);
  p[1] = static_cast<unsigned char>(g);
  p[2] = static_cast<unsigned char>(r);
}

}  // namespace g4gpu::vis
