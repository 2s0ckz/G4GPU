// The visualization manager: an interactive viewer in the spirit of Geant4's OpenGL viewer,
// with a run control bar. Declared in render/vis_manager.h and compiled once by build_vis.bat.
//
// Zero external dependencies: raw Win32 for the window and WGL for the GL context, both from
// the Windows SDK. No GLFW, no GLEW, no extension loading - the display path uses only GL 1.1
// (a texture and a quad), which ships in opengl32.dll.
//
// Every frame: CUDA renders the scene, the result is resolved into a host RGBA buffer, the
// immediate-mode UI is composited into that same buffer on the CPU, and the whole thing is
// uploaded as one texture and drawn as a fullscreen quad. Compositing the UI on the host is
// what keeps it dependency-free; it costs one pass over the panel pixels, which at 1280x800
// with a 300 px sidebar is about a quarter of a megapixel.
//
// The scene comes from the same G4 detector construction an example uses, so the viewer shows
// whatever the detector describes rather than a hard-coded B1.
//
// Controls
//   left drag    orbit                    SPACE   run the event count in the box
//   right drag   pan                      C       clear trajectories
//   wheel        zoom                      X      x-ray (tracks through solids)
//   R            reset camera              G      solids on/off
//   S            save a PNG                ESC    quit
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "g4/G4RunManager.hh"
#include "g4/G4UImanager.hh"
#include "host/transport_run.cuh"
#include "render/float_geometry.cuh"
#include "render/png.h"
#include "render/renderer.cuh"
#include "render/vis_manager.h"
#include "render/ui.h"
#include "scenes/scene_registry.hh"

using namespace g4gpu;
using real_t = double;

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    const cudaError_t err__ = (call);                                                    \
    if (err__ != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

// ---------------------------------------------------------------- wireframe edges

/// Wireframe for the volumes, so a transparent or hidden solid still reads as a shape.
struct EdgeList {
  std::vector<float> x0, y0, z0, x1, y1, z1;
  std::vector<unsigned int> rgb;

  void Add(float ax, float ay, float az, float bx, float by, float bz, unsigned int c) {
    x0.push_back(ax); y0.push_back(ay); z0.push_back(az);
    x1.push_back(bx); y1.push_back(by); z1.push_back(bz);
    rgb.push_back(c);
  }

  /// The twelve edges of an axis-aligned box, transformed by a volume's placement.
  void AddBox(const geom::Transform<real_t>& xf, real_t hx, real_t hy, real_t hz,
              unsigned int c) {
    const real_t sx[2] = {-hx, hx}, sy[2] = {-hy, hy}, sz[2] = {-hz, hz};
    auto pt = [&](real_t x, real_t y, real_t z) {
      // Local to world: the stored matrix is world -> local, so the inverse is its transpose.
      const g4gpu::Vec3<real_t> l{x, y, z};
      const real_t* r = xf.rot;
      return g4gpu::Vec3<real_t>{r[0] * l.x + r[3] * l.y + r[6] * l.z + xf.trans.x,
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

  std::size_t Size() const { return rgb.size(); }
};

// ---------------------------------------------------------------- application

constexpr int kSidebarW = 320;

struct App {
  HWND hwnd = nullptr;
  HDC hdc = nullptr;
  HGLRC hglrc = nullptr;
  GLuint tex = 0;
  int width = 1440, height = 880;
  bool running = true;
  bool resized = true;

  // orbit camera
  float azimuth = 0.55f, elevation = 0.32f, distance = 620.0f;
  vis::Vec3f target{0.f, 0.f, 0.f};
  float home_distance = 620.0f;
  bool orbiting = false, panning = false;
  int last_x = 0, last_y = 0;

  // display options
  bool xray = true;
  bool show_solids = true;
  bool show_wireframe = true;
  bool accumulate = true;

  // run control
  ui::NumberField events_field;
  int total_events = 0;
  int last_run_events = 0;
  double edep_total = 0;          ///< MeV in the first scorer, accumulated
  double last_ms = 0;
  int log_scroll = -1;
  std::vector<std::string> log;

  // device state
  host::TransportEngine<real_t>* engine = nullptr;
  vis::TrajectoryBuffer traj{};
  vis::VolumeStyle* d_styles = nullptr;
  /// The float copy of the scene the render pass walks. See render/float_geometry.cuh: the
  /// transport stays double because the dose depends on it, and the picture does not.
  vis::FloatGeometry render_geom;
  unsigned long long* d_fb = nullptr;
  unsigned int* d_rgba = nullptr;
  std::vector<unsigned int> host_rgba;
  int n_segments = 0;
  int dropped_segments = 0;

  float *ex0 = nullptr, *ey0 = nullptr, *ez0 = nullptr;
  float *ex1 = nullptr, *ey1 = nullptr, *ez1 = nullptr;
  unsigned int* ergb = nullptr;
  int n_edges = 0;

  ui::Font font, font_bold;
  ui::Context uic;
  ui::Input input;
};

static App g_app;

/// Appends a line to the on-screen log, and to stdout so a headless run still says what
/// happened.
static void Log(const std::string& s) {
  std::printf("%s\n", s.c_str());
  g_app.log.push_back(s);
  if (g_app.log.size() > 4000) {
    g_app.log.erase(g_app.log.begin(), g_app.log.begin() + 1000);
  }
  g_app.log_scroll = -1;  // pin to the end
}

static std::string Fmt(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

static vis::Camera CurrentCamera(const App& a) {
  const float el = std::max(-1.5f, std::min(1.5f, a.elevation));
  const int view_w = std::max(1, a.width - kSidebarW);
  // +Z IS UP; see the builder's CurrentCamera for why, and they have to agree - the two draw
  // the same scene and a screenshot from one is compared against the other by eye.
  const vis::Vec3f eye{a.target.x + a.distance * std::cos(el) * std::sin(a.azimuth),
                       a.target.y + a.distance * std::cos(el) * std::cos(a.azimuth),
                       a.target.z + a.distance * std::sin(el)};
  return vis::make_camera(eye, a.target, vis::Vec3f{0.f, 0.f, 1.f}, 45.0f, view_w, a.height);
}

// ---------------------------------------------------------------- transport

static void ClearTracks(App& a) {
  CUDA_CHECK(cudaMemset(a.traj.count, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(a.traj.dropped, 0, sizeof(int)));
  a.n_segments = 0;
  a.dropped_segments = 0;
  a.total_events = 0;
  a.edep_total = 0;
  Log("trajectories cleared");
}

/// Runs `n` events, appending trajectories. Resetting first is the default because the control
/// bar's Run button is meant to show "this run", not a growing pile.
static void BeamOn(App& a, int n) {
  if (n <= 0) { return; }
  auto* rm = G4RunManager::Instance();
  if (rm == nullptr || a.engine == nullptr) { return; }

  if (!a.accumulate) {
    CUDA_CHECK(cudaMemset(a.traj.count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(a.traj.dropped, 0, sizeof(int)));
    a.n_segments = 0;
    a.edep_total = 0;
    a.total_events = 0;
  }

  // Only the first few events are drawn: a thousand events is a solid block of color, and the
  // trajectory buffer would be exhausted by the first batch anyway.
  a.traj.max_event = std::min(n, 200);

  // Through the run manager, not around it.
  //
  // This used to read `gun->GetSource()` and call `engine->BeamOn` directly - a private copy
  // of what G4RunManager::BeamOn does - and the copy was missing the two things that matter.
  // It never called GeneratePrimaries, so the gun was never configured by the project's own
  // primary generator and stayed as constructed: a run in the viewer fired an unconfigured
  // gun, which draws as a point source spraying in every direction. And it knew nothing of
  // sources beyond the first, so a model with two beams ran as one.
  //
  // Both were reported from the viewer of a generated project, where the geometry was right
  // and the beam was not.
  std::vector<double> sums, sums_sq;
  const auto stats = rm->RunEvents(n, sums, sums_sq, a.traj);

  int seg = 0, dropped = 0;
  CUDA_CHECK(cudaMemcpy(&seg, a.traj.count, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&dropped, a.traj.dropped, sizeof(int), cudaMemcpyDeviceToHost));
  a.n_segments = std::min(seg, a.traj.capacity);
  a.dropped_segments = dropped;
  a.total_events += n;
  a.last_run_events = n;
  a.last_ms = stats.milliseconds;
  if (!sums.empty()) { a.edep_total += sums[0]; }

  Log(Fmt("/run/beamOn %d", n));
  Log(Fmt("  %.1f ms, %.3g events/s, %lld track-steps", stats.milliseconds,
          n / (stats.milliseconds * 1e-3), stats.track_steps));
  if (!sums.empty()) {
    const double mass = rm->ScoredMass(0);
    const double dose_pGy = (mass > 0) ? sums[0] * MeV / joule / mass * gray / picogray : 0.0;
    Log(Fmt("  edep %.6g MeV in scorer 0; dose %.4g pGy", sums[0], dose_pGy));
  }
  if (stats.abandoned > 0) {
    Log(Fmt("  WARNING: %lld tracks abandoned at the iteration limit", stats.abandoned));
  }
  if (dropped > 0) {
    Log(Fmt("  %d trajectory segments dropped (buffer holds %d)", dropped, a.traj.capacity));
  }
}

// ---------------------------------------------------------------- GL surface

static void AllocSurface(App& a) {
  if (a.d_fb != nullptr) { cudaFree(a.d_fb); }
  if (a.d_rgba != nullptr) { cudaFree(a.d_rgba); }
  const int view_w = std::max(1, a.width - kSidebarW);
  CUDA_CHECK(cudaMalloc(&a.d_fb, sizeof(unsigned long long) * view_w * a.height));
  CUDA_CHECK(cudaMalloc(&a.d_rgba, sizeof(unsigned int) * view_w * a.height));
  a.host_rgba.assign(static_cast<size_t>(a.width) * a.height, ui::theme::kPanel);

  if (a.tex == 0) { glGenTextures(1, &a.tex); }
  glBindTexture(GL_TEXTURE_2D, a.tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, a.width, a.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  a.resized = false;
}

// ---------------------------------------------------------------- the control panel

/// Draws the sidebar. Returns true if a run was requested this frame.
static bool DrawSidebar(App& a) {
  ui::Context& c = a.uic;
  const ui::Rect side{a.width - kSidebarW, 0, kSidebarW, a.height};
  ui::Panel(c, side);

  const int lh = a.font.glyph_h;
  int y = 10;
  y = ui::SectionHeader(c, side, y, "RUN CONTROL");

  // Event count and the Run / Reset pair.
  c.canvas.Text(side.x + 8, y + 4, "events", ui::theme::kTextDim);
  const ui::Rect field{side.x + 80, y, kSidebarW - 96, lh + 8};
  const bool submitted = ui::NumberInput(c, 1, field, a.events_field, "10000");
  y += lh + 16;

  const int bw = (kSidebarW - 24) / 2;
  const bool run = ui::Button(c, 2, {side.x + 8, y, bw, lh + 12}, "Run") || submitted;
  const bool reset = ui::Button(c, 3, {side.x + 16 + bw, y, bw, lh + 12}, "Reset");
  y += lh + 22;

  ui::Checkbox(c, 4, {side.x + 8, y, kSidebarW - 16, lh + 4}, "accumulate across runs",
               a.accumulate);
  y += lh + 10;

  // What the last run produced.
  y = ui::SectionHeader(c, side, y + 6, "RESULTS");
  auto* rm = G4RunManager::Instance();
  const double mass = (rm != nullptr) ? rm->ScoredMass(0) : 0.0;
  const double dose_pGy =
      (mass > 0) ? a.edep_total * MeV / joule / mass * gray / picogray : 0.0;
  c.canvas.Text(side.x + 8, y, Fmt("events        %d", a.total_events), ui::theme::kText);
  y += lh + 2;
  c.canvas.Text(side.x + 8, y, Fmt("edep          %.6g MeV", a.edep_total), ui::theme::kText);
  y += lh + 2;
  c.canvas.Text(side.x + 8, y, Fmt("scorer mass   %.4f kg", mass), ui::theme::kTextDim);
  y += lh + 2;
  c.canvas.Text(side.x + 8, y, Fmt("dose          %.5g pGy", dose_pGy), ui::theme::kOk);
  y += lh + 2;
  if (a.last_ms > 0) {
    c.canvas.Text(side.x + 8, y,
                  Fmt("rate          %.3g events/s", a.last_run_events / (a.last_ms * 1e-3)),
                  ui::theme::kTextDim);
  }
  y += lh + 6;

  // Display toggles.
  y = ui::SectionHeader(c, side, y, "DISPLAY");
  ui::Checkbox(c, 10, {side.x + 8, y, kSidebarW - 16, lh + 4}, "solids (G)", a.show_solids);
  y += lh + 8;
  ui::Checkbox(c, 11, {side.x + 8, y, kSidebarW - 16, lh + 4}, "wireframe", a.show_wireframe);
  y += lh + 8;
  ui::Checkbox(c, 12, {side.x + 8, y, kSidebarW - 16, lh + 4}, "x-ray tracks (X)", a.xray);
  y += lh + 14;

  // The geometry, so the user can see what is loaded.
  y = ui::SectionHeader(c, side, y, "VOLUMES");
  if (rm != nullptr) {
    const auto& scene = rm->GetScene();
    for (std::size_t i = 0; i < scene.volumes.size() && y < a.height - 220; ++i) {
      const auto& st = scene.styles[i];
      const ui::Color swatch = ui::rgb(static_cast<int>(st.r * 255),
                                        static_cast<int>(st.g * 255),
                                        static_cast<int>(st.b * 255));
      const std::string label = Fmt("%-12s L%d %s", scene.names[i].c_str(),
                                    scene.volumes[i].layer,
                                    scene.volumes[i].score_index >= 0 ? "[scored]" : "");
      ui::ListRow(c, 100 + static_cast<int>(i), {side.x + 8, y, kSidebarW - 16, lh + 4}, label,
                  false, swatch);
      y += lh + 4;
    }
  }

  // The log fills whatever is left.
  const int log_top = std::max(y + 10, a.height - 210);
  ui::SectionHeader(c, side, log_top, "OUTPUT");
  const ui::Rect log_r{side.x + 8, log_top + lh + 8, kSidebarW - 16,
                       a.height - log_top - lh - 18};
  ui::TextLog(c, log_r, a.log, a.log_scroll);

  if (reset) { ClearTracks(a); }
  return run;
}

/// Copies the current frame out of the RGBA surface and writes it as a PNG.
///
/// The RGBA surface is what the renderer wrote and what the window shows, so this is the
/// picture on screen and not a second rendering of it - there is no separate offscreen path
/// to get out of step with the interactive one. The PNG writer wants 24-bit RGB, so alpha is
/// dropped on the way out.
/// How many pixels the *solid* pass covered in the frame just drawn.
///
/// Read from the depth-and-colour buffer rather than from the finished image, because the
/// finished image cannot tell the passes apart: wireframe edges, trajectories and the
/// background all land in the same RGBA, and a picture made entirely of lines looks like a
/// deliberate wireframe view rather than like solid geometry that failed to draw.
///
/// That distinction is the whole reason this exists. `VolumeStyle` grew an alpha channel for
/// transparency; the viewer's style setup was not updated to fill it in; `std::vector` zeroed
/// it; and `render_geometry` skips a volume whose alpha is zero. Every solid in the viewer
/// stopped rendering, and nothing noticed - the selftest saved a PNG and checked that it had
/// been written, the offscreen pictures were compared for existence, and the trajectory count
/// was non-zero because the line passes were unaffected.
static int CountSolidPixels(App& a) {
  const int view_w = std::max(1, a.width - kSidebarW);
  if (a.d_fb == nullptr || a.height <= 0) { return 0; }
  std::vector<unsigned long long> fb(static_cast<std::size_t>(view_w) * a.height);
  CUDA_CHECK(cudaMemcpy(fb.data(), a.d_fb, sizeof(unsigned long long) * fb.size(),
                        cudaMemcpyDeviceToHost));
  int n = 0;
  for (unsigned long long v : fb) {
    if (v != vis::kEmptyPixel) { ++n; }
  }
  return n;
}
static bool SaveFrame(App& a, const std::string& path) {
  if (a.host_rgba.empty() || a.width <= 0 || a.height <= 0) {
    Log("nothing rendered yet - no frame to save");
    return false;
  }
  std::vector<unsigned char> rgb(static_cast<size_t>(a.width) * a.height * 3);
  for (size_t i = 0; i < static_cast<size_t>(a.width) * a.height; ++i) {
    const unsigned p = a.host_rgba[i];
    rgb[i * 3 + 0] = static_cast<unsigned char>(p & 255);
    rgb[i * 3 + 1] = static_cast<unsigned char>((p >> 8) & 255);
    rgb[i * 3 + 2] = static_cast<unsigned char>((p >> 16) & 255);
  }
  if (!vis::write_png_rgb(path.c_str(), rgb.data(), a.width, a.height)) {
    Log("could not write " + path);
    return false;
  }
  Log("saved " + path + " (" + std::to_string(a.width) + "x" + std::to_string(a.height) + ")");
  return true;
}

/// One frame: CUDA render into the viewport region, composite the UI, upload, draw.
static void DrawFrame(App& a) {
  if (a.resized) { AllocSurface(a); }
  const int view_w = std::max(1, a.width - kSidebarW);

  auto* rm = G4RunManager::Instance();
  const vis::Camera cam = CurrentCamera(a);
  const dim3 block(16, 16);
  const dim3 grid((view_w + 15) / 16, (a.height + 15) / 16);

  if (a.show_solids && rm != nullptr) {
    vis::render_geometry<float><<<grid, block>>>(a.render_geom.geometry(), a.d_styles, cam,
                                                 a.d_fb);
  } else {
    CUDA_CHECK(cudaMemset(a.d_fb, 0xFF, sizeof(unsigned long long) * view_w * a.height));
  }
  if (a.show_wireframe && a.n_edges > 0) {
    vis::render_edges<<<(a.n_edges + 63) / 64, 64>>>(a.ex0, a.ey0, a.ez0, a.ex1, a.ey1, a.ez1,
                                                     a.ergb, a.n_edges, cam, a.d_fb, 0);
  }
  if (a.n_segments > 0) {
    vis::render_trajectories<<<(a.n_segments + 127) / 128, 128>>>(
        a.traj, a.n_segments, cam, a.d_fb, 0, a.xray ? 1e-3f : 1.0f);
  }
  vis::resolve_to_rgba<<<grid, block>>>(a.d_fb, a.d_rgba, view_w, a.height);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  // Copy the render into the left part of the host buffer, row by row, because the host buffer
  // is `width` wide and the render is `view_w`.
  for (int y = 0; y < a.height; ++y) {
    CUDA_CHECK(cudaMemcpy(&a.host_rgba[static_cast<size_t>(y) * a.width],
                          a.d_rgba + static_cast<size_t>(y) * view_w,
                          sizeof(unsigned int) * view_w, cudaMemcpyDeviceToHost));
  }

  // UI over the top.
  a.uic.Begin(a.host_rgba.data(), a.width, a.height, &a.font, &a.input);
  const bool run = DrawSidebar(a);
  a.uic.End();
  if (run) {
    BeamOn(a, static_cast<int>(a.events_field.value));
  }

  glViewport(0, 0, a.width, a.height);
  glBindTexture(GL_TEXTURE_2D, a.tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.width, a.height, GL_RGBA, GL_UNSIGNED_BYTE,
                  a.host_rgba.data());

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 1); glVertex2f(0, 0);
  glTexCoord2f(1, 1); glVertex2f(1, 0);
  glTexCoord2f(1, 0); glVertex2f(1, 1);
  glTexCoord2f(0, 0); glVertex2f(0, 1);
  glEnd();
  glDisable(GL_TEXTURE_2D);
  SwapBuffers(a.hdc);
  a.input.EndFrame();
}

// ---------------------------------------------------------------- input

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  App& a = g_app;
  const bool over_view = a.input.mouse_x < a.width - kSidebarW;
  switch (msg) {
    case WM_CLOSE:
    case WM_DESTROY:
      a.running = false;
      return 0;
    case WM_SIZE: {
      const int w = LOWORD(lp), h = HIWORD(lp);
      if (w > 0 && h > 0 && (w != a.width || h != a.height)) {
        a.width = w;
        a.height = h;
        a.resized = true;
      }
      return 0;
    }
    case WM_LBUTTONDOWN:
      a.input.left_down = true;
      a.input.left_pressed = true;
      if (over_view) {
        a.orbiting = true;
        a.last_x = static_cast<short>(LOWORD(lp));
        a.last_y = static_cast<short>(HIWORD(lp));
      }
      SetCapture(hwnd);
      return 0;
    case WM_RBUTTONDOWN:
      a.input.right_down = true;
      a.input.right_pressed = true;
      if (over_view) {
        a.panning = true;
        a.last_x = static_cast<short>(LOWORD(lp));
        a.last_y = static_cast<short>(HIWORD(lp));
      }
      SetCapture(hwnd);
      return 0;
    case WM_LBUTTONUP:
      a.input.left_down = false;
      a.input.left_released = true;
      a.orbiting = false;
      ReleaseCapture();
      return 0;
    case WM_RBUTTONUP:
      a.input.right_down = false;
      a.input.right_released = true;
      a.panning = false;
      ReleaseCapture();
      return 0;
    case WM_MOUSEMOVE: {
      const int x = static_cast<short>(LOWORD(lp)), y = static_cast<short>(HIWORD(lp));
      const int dx = x - a.last_x, dy = y - a.last_y;
      a.input.mouse_x = x;
      a.input.mouse_y = y;
      if (a.orbiting) {
        a.azimuth -= dx * 0.008f;
        a.elevation += dy * 0.008f;
      } else if (a.panning) {
        const vis::Camera cc = CurrentCamera(a);
        const float s = a.distance * 0.0015f;
        a.target = a.target - (dx * s) * cc.right + (dy * s) * cc.up;
      }
      a.last_x = x;
      a.last_y = y;
      return 0;
    }
    case WM_MOUSEWHEEL: {
      const int delta = GET_WHEEL_DELTA_WPARAM(wp);
      a.input.wheel += delta / 120;
      if (a.input.mouse_x < a.width - kSidebarW) {
        a.distance *= std::pow(0.9f, delta / 120.0f);
        a.distance = std::max(a.home_distance * 0.05f,
                              std::min(a.home_distance * 8.0f, a.distance));
      }
      return 0;
    }
    case WM_CHAR:
      if (wp >= 8 && wp < 127) { a.input.typed.push_back(static_cast<char>(wp)); }
      return 0;
    case WM_KEYDOWN: {
      a.input.keys.push_back(static_cast<int>(wp));
      a.input.ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
      a.input.shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
      // Shortcuts are suppressed while a text field has focus, so typing "R" into the event
      // count does not also reset the camera.
      if (a.uic.focus != 0) { return 0; }
      switch (wp) {
        case VK_ESCAPE: a.running = false; break;
        case VK_SPACE:  BeamOn(a, static_cast<int>(a.events_field.value)); break;
        case 'C':       ClearTracks(a); break;
        case 'X':       a.xray = !a.xray; break;
        case 'G':       a.show_solids = !a.show_solids; break;
        case 'W':       a.show_wireframe = !a.show_wireframe; break;
        case 'R':
          a.azimuth = 0.55f;
          a.elevation = 0.32f;
          a.distance = a.home_distance;
          a.target = vis::Vec3f{0.f, 0.f, 0.f};
          break;
        case 'S': {
          // Relative, and next to the executable's working directory rather than an absolute
          // D:\g4gpu path: the viewer is linked into examples and generated projects, which do
          // not run from this tree.
          SaveFrame(a, "g4view.png");
          break;
        }
        default: break;
      }
      return 0;
    }
    default:
      break;
  }
  return DefWindowProc(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------- vis commands

/// Handles the /vis/ subset a Geant4 visualization macro uses. Anything recognised changes the
/// viewer; anything else is reported so that a macro typo is visible rather than silent.
// ------------------------------------------------------------------ offscreen export
//
// Geant4's B1 ships tsg_offscreen.mac, which drives the ToolsSG offscreen viewer: a file
// name, a format, a size, and then a /vis/viewer/rebuild to produce a picture. There is no
// ToolsSG here, but there is a renderer and a PNG writer, so the command set is honoured for
// the format it can honour and refused with a reason for the rest.
//
// What is supported:
//   /vis/ogl/export [file] [w] [h]        write the current frame; the OpenGL driver's name
//                                         for this, and the one most macros use
//   /vis/tsg/offscreen/set/file <name>    arm a file; auto <prefix> [reset] numbers them
//   /vis/tsg/offscreen/set/format zb_png  the only format here
//   /vis/viewer/rebuild                   redraw, and write the armed file if there is one
//
// What is refused, and why: the gl2ps vector formats (eps, ps, pdf, svg, tex, pgf) would need
// a vector back end this renderer does not have - it rasterises on the GPU and has no display
// list to replay - and jpeg would need an encoder. A size other than the window's is refused
// because the surface is allocated at Open() and the camera's aspect follows it; -w/-h on the
// command line, or the geometry string to /vis/open, set it before that happens.
static std::string g_offscreen_file;   ///< armed by /vis/tsg/offscreen/set/file
static std::string g_offscreen_prefix; ///< set by `... set/file auto <prefix>`
static int g_offscreen_index = 0;

/// The next file name, and bump the index if the name is auto-generated.
static std::string NextOffscreenFile() {
  if (g_offscreen_prefix.empty()) { return g_offscreen_file; }
  ++g_offscreen_index;
  return g_offscreen_prefix + std::to_string(g_offscreen_index) + ".png";
}

/// Commands that change nothing in this viewer, each for a reason:
///
///   autoRefresh            this viewer redraws every frame; there is nothing to defer.
///   verbose                the output panel shows everything; there is no level to set.
///   auxiliaryEdge          the wireframe pass draws box edges unconditionally.
///   lineSegmentsPerCircle  curved solids are ray cast, not tessellated, so a circle has no
///                          segment count. This is the one place where "no effect" is a
///                          better picture rather than a worse one.
///   hiddenEdge, culling, projection, upVector, lightsMove, defaultColour, globalLineWidth
///                          renderer settings this renderer does not have.
///
/// They are listed rather than swept up by a prefix match, because B1's own vis.mac uses four
/// of them and an unhandled /vis/ command makes the macro - and therefore the example - report
/// failure. A stock macro has to run clean; anything genuinely unimplemented should still be
/// reported, which is what the fall-through at the end of HandleVisCommand does.
///
/// This is a separate function, and not just a branch inside HandleVisCommand, because none of
/// these needs a window and ApplyVisCommand opens one for anything it does not recognise. So
/// `/vis/verbose confirmations` on the line *before* `/vis/open` opened the viewer at the
/// default size, and the real `/vis/open TSG_OFFSCREEN 1200x1200` then found it already open
/// and returned true - rendering at 1440x880 with no complaint. Geant4's own vis.mac starts
/// with /vis/open, which is why that went unnoticed.
static bool IsNoEffectVisCommand(const std::string& cmd) {
  const std::size_t e = cmd.find_first_of(" \t");
  const std::string h = cmd.substr(0, (e == std::string::npos) ? cmd.size() : e);
  return h == "/vis/viewer/set/autoRefresh" || h == "/vis/verbose"
         || h == "/vis/viewer/set/auxiliaryEdge" || h == "/vis/viewer/set/lineSegmentsPerCircle"
         || h == "/vis/viewer/set/hiddenEdge" || h == "/vis/viewer/set/culling"
         || h == "/vis/viewer/set/projection" || h == "/vis/viewer/set/upVector"
         || h == "/vis/viewer/set/lightsMove" || h == "/vis/viewer/set/defaultColour"
         || h == "/vis/viewer/set/globalLineWidth" || h == "/vis/viewer/set/lightsVector";
}

static bool HandleVisCommand(const std::string& cmd) {
  App& a = g_app;
  std::vector<std::string> tok;
  {
    std::size_t i = 0;
    while (i < cmd.size()) {
      while (i < cmd.size() && std::isspace(static_cast<unsigned char>(cmd[i]))) { ++i; }
      const std::size_t s = i;
      while (i < cmd.size() && !std::isspace(static_cast<unsigned char>(cmd[i]))) { ++i; }
      if (i > s) { tok.push_back(cmd.substr(s, i - s)); }
    }
  }
  if (tok.empty()) { return true; }
  const std::string& h = tok[0];
  auto num = [&](std::size_t i, double def = 0.0) {
    return (i < tok.size()) ? std::atof(tok[i].c_str()) : def;
  };

  // ------------------------------------------------------------------ offscreen export
  //
  // Geant4's B1 ships tsg_offscreen.mac, which drives the ToolsSG offscreen viewer: arm a file
  // name, pick a format, then /vis/viewer/rebuild to produce a picture. There is no ToolsSG
  // here, but there is a renderer and a PNG writer, so the command set is honoured for the one
  // format it can honour and refused with a reason for the rest.
  //
  // The gl2ps vector formats (eps, ps, pdf, svg, tex, pgf) would need a vector back end this
  // renderer does not have - it rasterises on the GPU and keeps no display list to replay -
  // and jpeg would need an encoder. A size other than the window's is refused because the
  // surface is allocated at Open() and the camera's aspect follows it; set it before that, with
  // -w/-h or the geometry string to /vis/open.
  if (h == "/vis/ogl/export" || h == "/vis/tsg/offscreen/export") {
    DrawFrame(a);
    const std::string file = (tok.size() > 1) ? tok[1] : NextOffscreenFile();
    return SaveFrame(a, file.empty() ? std::string("g4gpu_export.png") : file);
  }
  if (h == "/vis/tsg/offscreen/set/file") {
    if (tok.size() > 1 && tok[1] == "auto") {
      g_offscreen_prefix = (tok.size() > 2) ? tok[2] : std::string("g4gpu_offscreen_");
      const bool reset = (tok.size() > 3) && (tok[3] == "true" || tok[3] == "1");
      if (reset) { g_offscreen_index = 0; }
      g_offscreen_file.clear();
      Log("offscreen files will be named " + g_offscreen_prefix + "<n>.png");
    } else if (tok.size() > 1) {
      g_offscreen_file = tok[1];
      g_offscreen_prefix.clear();
      Log("offscreen file: " + g_offscreen_file);
    }
    return true;
  }
  if (h == "/vis/tsg/offscreen/set/format") {
    const std::string fmt = (tok.size() > 1) ? tok[1] : std::string("zb_png");
    if (fmt == "zb_png") { return true; }
    Log("format " + fmt + " is not available here; only zb_png is.");
    Log("  The gl2ps formats need a vector back end this renderer does not have, and jpeg");
    Log("  needs an encoder. Nothing is written rather than a PNG with the wrong suffix.");
    return false;
  }
  if (h == "/vis/tsg/offscreen/set/size") {
    const int w = static_cast<int>(num(1, 0)), ht = static_cast<int>(num(2, 0));
    if ((w == 0 && ht == 0) || (w == a.width && ht == a.height)) {
      return true;  // "0 0" means "back to the viewer's own size", which is all there is
    }
    Log(Fmt("cannot render at %dx%d: this viewer's surface is allocated at /vis/open and the",
            w, ht));
    Log(Fmt("  current size is %dx%d. Use /vis/open <driver> %dx%d, or -w/-h, before opening.",
            a.width, a.height, w, ht));
    return false;
  }
  if (h == "/vis/tsg/offscreen/set/transparency") {
    return true;  // this renderer's transparency is not format-dependent, so nothing to do
  }
  if (h == "/vis/viewer/rebuild") {
    // A rebuild redraws, and writes the armed file if one is armed - which is how
    // tsg_offscreen.mac produces its pictures.
    DrawFrame(a);
    if (!g_offscreen_file.empty() || !g_offscreen_prefix.empty()) {
      return SaveFrame(a, NextOffscreenFile());
    }
    return true;
  }


  if (h == "/vis/open" || h == "/vis/drawVolume" || h == "/vis/scene/create"
      || h == "/vis/sceneHandler/attach" || h == "/vis/scene/add/trajectories"
      || h == "/vis/scene/add/axes" || h == "/vis/scene/add/scale"
      || h == "/vis/modeling/trajectories/create/drawByParticleID"
      || h == "/vis/viewer/flush") {
    return true;  // the viewer is already open and already drawing these
  }
  // Accepted and without effect here; see IsNoEffectVisCommand for what and why. Kept as a
  // branch as well so that HandleVisCommand alone is a complete answer to "is this handled".
  if (IsNoEffectVisCommand(cmd)) { return true; }
  if (h == "/vis/scene/endOfEventAction") {
    a.accumulate = (tok.size() > 1 && tok[1] == "accumulate");
    return true;
  }
  if (h == "/vis/viewer/set/viewpointThetaPhi") {
    a.elevation = static_cast<float>((90.0 - num(1)) * 3.14159265358979 / 180.0);
    a.azimuth = static_cast<float>(num(2) * 3.14159265358979 / 180.0);
    return true;
  }
  if (h == "/vis/viewer/zoom" || h == "/vis/viewer/zoomTo") {
    const double z = num(1, 1.0);
    if (z > 0) { a.distance = static_cast<float>(a.home_distance / z); }
    return true;
  }
  if (h == "/vis/viewer/set/style") {
    a.show_solids = (tok.size() > 1 && (tok[1] == "surface" || tok[1] == "s"));
    return true;
  }
  if (h == "/vis/viewer/set/background") {
    return true;  // the background is a fixed gradient in the resolve kernel
  }
  if (h == "/vis/geometry/set/visibility") {
    // /vis/geometry/set/visibility <logical> <depth> <flag>
    if (tok.size() < 4) { return false; }
    auto* rm = G4RunManager::Instance();
    if (rm == nullptr) { return false; }
    const bool on = (tok[3] == "true" || tok[3] == "1");
    const auto& scene = rm->GetScene();
    std::vector<vis::VolumeStyle> styles(scene.volumes.size());
    for (std::size_t i = 0; i < scene.volumes.size(); ++i) {
      const auto& st = scene.styles[i];
      styles[i].r = static_cast<unsigned char>(st.r * 255);
      styles[i].g = static_cast<unsigned char>(st.g * 255);
      styles[i].b = static_cast<unsigned char>(st.b * 255);
      styles[i].a = static_cast<unsigned char>(st.opacity * 255 + 0.5f);
      styles[i].solid = st.visible && !st.wireframe;
      if (scene.names[i] == tok[1]) { styles[i].solid = on; }
    }
    CUDA_CHECK(cudaMemcpy(a.d_styles, styles.data(),
                          sizeof(vis::VolumeStyle) * styles.size(), cudaMemcpyHostToDevice));
    return true;
  }
  if (h.rfind("/vis/modeling/trajectories/", 0) == 0) {
    return true;  // color by species is what the renderer already does
  }
  Log("unhandled vis command: " + cmd);
  return false;
}

// ---------------------------------------------------------------- entry points

namespace g4gpu::vis::viewer {
namespace {

Options g_opt;
bool g_open = false;
bool g_open_failed = false;  ///< latched, so a macro reports the reason once, not per command
int g_frame = 0;

/// Parses Geant4's /vis/open geometry string - "1280x800", or "600x600-0+0" with a window
/// position this viewer ignores. Leaves the size unchanged if there is no WxH in it.
void ParseGeometry(const std::string& g, int& w, int& h) {
  const std::size_t x = g.find('x');
  if (x == std::string::npos || x == 0) { return; }
  const int pw = std::atoi(g.substr(0, x).c_str());
  int i = static_cast<int>(x) + 1;
  std::string digits;
  while (i < static_cast<int>(g.size()) && std::isdigit(static_cast<unsigned char>(g[i]))) {
    digits.push_back(g[i]);
    ++i;
  }
  const int ph = std::atoi(digits.c_str());
  if (pw > 0 && ph > 0) { w = pw; h = ph; }
}

}  // namespace

void LogLine(const std::string& s) { Log(s); }

void BeamOnEvents(int n) { ::BeamOn(g_app, n); }

const Options& CurrentOptions() { return g_opt; }

bool IsOpen() { return g_open; }

bool Open(const Options& opt) {
  if (g_open) { return true; }
  App& a = g_app;
  g_opt = opt;
  a.width = opt.width;
  a.height = opt.height;

  auto* rm = G4RunManager::Instance();
  if (rm == nullptr) {
    std::printf("ERROR: /vis/open with no run manager.\n");
    g_open_failed = true;
    return false;
  }
  if (rm->GetScene().volumes.empty()) {
    std::printf(
        "ERROR: /vis/open before /run/initialize - there is no geometry to draw.\n"
        "  Geant4's ordering is /run/initialize first and the visualization macro second,\n"
        "  which is what init_vis.mac does. Run the example with no arguments, or with\n"
        "  init_vis.mac, rather than vis.mac on its own.\n");
    g_open_failed = true;
    return false;
  }
  a.engine = &rm->GetEngine();
  const auto& scene = rm->GetScene();

  // Size the camera from the world's extent, so any detector frames sensibly.
  {
    double reach = 0;
    for (const auto& v : scene.volumes) {
      reach = std::max(reach, geom::solid_half_extent(scene.pool.store(), v.solid)
                                  + std::sqrt(dot(v.xform.trans, v.xform.trans)));
    }
    a.home_distance = static_cast<float>(std::max(50.0, reach * 3.0));
    a.distance = a.home_distance;
  }

  // ---- window + GL context, no external libraries
  WNDCLASSA wc{};
  wc.style = CS_OWNDC;
  wc.lpfnWndProc = WndProc;
  wc.hInstance = GetModuleHandle(nullptr);
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.lpszClassName = "g4gpu_view";
  RegisterClassA(&wc);

  RECT r{0, 0, a.width, a.height};
  AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
  a.hwnd = CreateWindowA("g4gpu_view", opt.title.c_str(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                         CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                         wc.hInstance, nullptr);
  if (a.hwnd == nullptr) {
    std::printf("FATAL: CreateWindow failed\n");
    return false;
  }
  a.hdc = GetDC(a.hwnd);

  PIXELFORMATDESCRIPTOR pfd{};
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;
  const int pf = ChoosePixelFormat(a.hdc, &pfd);
  SetPixelFormat(a.hdc, pf, &pfd);
  a.hglrc = wglCreateContext(a.hdc);
  if (a.hglrc == nullptr || !wglMakeCurrent(a.hdc, a.hglrc)) {
    std::printf("FATAL: wglCreateContext failed\n");
    return false;
  }
  ShowWindow(a.hwnd, SW_SHOW);

  // ---- fonts. Consolas is present on every supported Windows; the fallbacks are for the
  // case where it is not, and a failure here is fatal because a UI with no text is unusable.
  if (!a.font.Build("Consolas", 15) && !a.font.Build("Courier New", 15)
      && !a.font.Build("Lucida Console", 15)) {
    std::printf("FATAL: could not build a font atlas\n");
    return false;
  }
  a.font_bold.Build("Consolas", 15, true);

  // ---- volume styles and wireframe
  {
    std::vector<vis::VolumeStyle> styles(scene.volumes.size());
    EdgeList edges;
    for (std::size_t i = 0; i < scene.volumes.size(); ++i) {
      const auto& st = scene.styles[i];
      styles[i].r = static_cast<unsigned char>(st.r * 255);
      styles[i].g = static_cast<unsigned char>(st.g * 255);
      styles[i].b = static_cast<unsigned char>(st.b * 255);
      // The alpha. Missing here for as long as the field existed, and because std::vector
      // zero-initialises, every volume came out fully transparent - render_geometry skips a
      // volume whose alpha is zero, so the viewer drew no solid geometry at all. The
      // wireframe pass was unaffected, which is why it looked like a wireframe view.
      styles[i].a = static_cast<unsigned char>(st.opacity * 255 + 0.5f);
      // The world is left as wireframe: a solid world hides everything inside it.
      styles[i].solid = st.visible && !st.wireframe && static_cast<int>(i) != scene.world;
      // Wireframe only for boxes, using their real half-lengths. A bounding cube drawn round
      // a cone or a sphere is not a hint about its shape, it is a lie about it - and the
      // curved solids are ray-cast as surfaces anyway, so they need no outline.
      const auto& v = scene.volumes[i];
      if (v.solid.type == geom::SolidType::kBox) {
        edges.AddBox(v.xform, v.solid.p[0], v.solid.p[1], v.solid.p[2],
                     ui::rgb(static_cast<int>(st.r * 160), static_cast<int>(st.g * 160),
                             static_cast<int>(st.b * 160)));
      }
    }
    CUDA_CHECK(cudaMalloc(&a.d_styles, sizeof(vis::VolumeStyle) * styles.size()));
    CUDA_CHECK(cudaMemcpy(a.d_styles, styles.data(),
                          sizeof(vis::VolumeStyle) * styles.size(), cudaMemcpyHostToDevice));

    // The float copy the render pass walks, from the HOST pools rather than the device ones
    // the engine uploaded - the conversion is arithmetic and the host is where the doubles
    // are. The voxel arrays are handed over as they stand: cells are shorts and class layers
    // are ints, so both precisions read the same bytes.
    {
      vis::HostGeometry hg{};
      hg.volumes = scene.volumes.data();
      hg.n_volumes = static_cast<int>(scene.volumes.size());
      hg.world = scene.world;
      hg.solids = scene.pool.solids.data();
      hg.n_solids = static_cast<int>(scene.pool.solids.size());
      hg.xforms = scene.pool.xforms.data();
      hg.n_xforms = static_cast<int>(scene.pool.xforms.size());
      hg.aux = scene.pool.aux.data();
      hg.n_aux = static_cast<int>(scene.pool.aux.size());
      hg.tri = scene.pool.tri.data();
      hg.n_tri = static_cast<int>(scene.pool.tri.size());
      hg.bvh = scene.pool.bvh.data();
      hg.n_bvh = static_cast<int>(scene.pool.bvh.size());
      const auto& gd = a.engine->geometry();
      geom::VoxelStore<float> vf{};
      vf.material = gd.voxels.material;
      vf.count = gd.voxels.count;
      vf.cls = gd.voxels.cls;
      vf.class_layer = gd.voxels.class_layer;
      a.render_geom.Build(hg, vf);
    }

    a.n_edges = static_cast<int>(edges.Size());
    const size_t nb = sizeof(float) * a.n_edges;
    CUDA_CHECK(cudaMalloc(&a.ex0, nb)); CUDA_CHECK(cudaMalloc(&a.ey0, nb));
    CUDA_CHECK(cudaMalloc(&a.ez0, nb)); CUDA_CHECK(cudaMalloc(&a.ex1, nb));
    CUDA_CHECK(cudaMalloc(&a.ey1, nb)); CUDA_CHECK(cudaMalloc(&a.ez1, nb));
    CUDA_CHECK(cudaMalloc(&a.ergb, sizeof(unsigned int) * a.n_edges));
    CUDA_CHECK(cudaMemcpy(a.ex0, edges.x0.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ey0, edges.y0.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ez0, edges.z0.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ex1, edges.x1.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ey1, edges.y1.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ez1, edges.z1.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.ergb, edges.rgb.data(), sizeof(unsigned int) * a.n_edges,
                          cudaMemcpyHostToDevice));
  }

  // ---- trajectory buffer
  {
    CUDA_CHECK(vis::allocate_trajectory(a.traj, 4 << 20, /*max_event=*/200));
  }

  a.events_field.Init(opt.initial_events > 0 ? opt.initial_events : 1000, "%.0f");

  // The UI log doubles as the macro echo, so a macro run shows in the panel.
  auto* uim = G4UImanager::GetUIpointer();
  uim->SetVisHandler(ApplyVisCommand);
  uim->SetOutputSink([](const G4String& s) { g_app.log.push_back(s); });

  if (opt.fixed_view) {
    a.elevation = static_cast<float>((90.0 - opt.shot_theta) * 3.14159265358979 / 180.0);
    a.azimuth = static_cast<float>(opt.shot_phi * 3.14159265358979 / 180.0);
  }

  g_open = true;
  g_frame = 0;
  Log(Fmt("viewer open: %d volumes, %d materials, %dx%d",
          static_cast<int>(scene.volumes.size()), scene.materials.count, a.width, a.height));
  // Windows will not give a client area larger than the screen's, so a /vis/open geometry
  // string can be honoured only in part. Said out loud, because a picture silently rendered
  // at 1200x1061 when 1200x1200 was asked for is the kind of thing someone measures against.
  if (a.width != opt.width || a.height != opt.height) {
    Log(Fmt("  asked for %dx%d; the window manager gave %dx%d (the screen is the limit)",
            opt.width, opt.height, a.width, a.height));
  }
  Log("SPACE runs the event count in the box; drag to orbit, wheel to zoom");
  return true;
}

int Loop() {
  if (!g_open) {
    std::printf("FATAL: viewer loop with no window; /vis/open first.\n");
    return 2;
  }
  App& a = g_app;
  while (a.running) {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) { a.running = false; }
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    if (!a.running) { break; }

    if (g_opt.selftest_frames > 0) {
      // Drive the camera and one run programmatically, then save and exit.
      //
      // -shot holds a fixed viewpoint instead of orbiting, which is what makes the saved
      // image comparable between runs - an orbiting camera gives a different frame every
      // time and cannot be used to check that a solid still looks like itself.
      if (!g_opt.fixed_view) {
        a.azimuth += 0.02f;
        if (g_frame == 10) { ::BeamOn(a, 200); }
        if (g_frame == 60) { a.xray = false; }
      }

      // Does the *solid* pass draw anything?
      //
      // Everything else this selftest checks is satisfied by a picture made entirely of
      // lines: the PNG is written, the trajectory count is non-zero, the offscreen files
      // exist. So when VolumeStyle grew an alpha channel and this file's style setup was not
      // updated to fill it in - leaving every volume at alpha zero, which render_geometry
      // skips - the viewer stopped drawing solid geometry entirely and every check passed.
      //
      // Turning the line passes off and counting what the solid pass covered is the one
      // measurement that tells "solid geometry rendered" from "the picture is made of lines".
      if (g_frame == g_opt.selftest_frames - 2) {
        a.show_wireframe = false;
        a.n_segments = 0;
      }
      if (g_frame == g_opt.selftest_frames - 1) {
        const int solid_px = CountSolidPixels(a);
        // A per-mille of the viewport. The selftest scene fills a good fraction of the frame,
        // so this is far below what a correct render produces and far above what stray
        // rounding could leave; the failure it guards against is exactly zero.
        const int floor_px = std::max(1, (a.width - kSidebarW) * a.height / 1000);
        if (solid_px < floor_px) {
          std::printf("selftest: FAILED - the solid pass covered %d pixels of %d; solid\n"
                      "  geometry is not rendering. Check VolumeStyle::a, which render_geometry\n"
                      "  treats as 'skip this volume' when it is zero.\n",
                      solid_px, (a.width - kSidebarW) * a.height);
        } else {
          std::printf("selftest: solid geometry rendered (%d pixels)\n", solid_px);
        }
        a.show_wireframe = true;
      }
      if (g_frame >= g_opt.selftest_frames) {
        SaveFrame(a, g_opt.png_path);
        std::printf("selftest: %d frames, %d segments, saved %s\n", g_frame, a.n_segments,
                    g_opt.png_path.c_str());
        break;
      }
    }
    DrawFrame(a);
    ++g_frame;
  }

  wglMakeCurrent(nullptr, nullptr);
  wglDeleteContext(a.hglrc);
  ReleaseDC(a.hwnd, a.hdc);
  DestroyWindow(a.hwnd);
  g_open = false;
  return 0;
}

bool ApplyVisCommand(const std::string& cmd) {
  // /vis/open [<driver>] [<geometry>] opens the window, as it does in Geant4. Everything else
  // needs a window to act on, so an unopened viewer opens one at the default size rather than
  // silently dropping the rest of the macro.
  if (cmd.rfind("/vis/open", 0) == 0) {
    Options opt = g_opt;
    std::size_t sp = cmd.find(' ');
    while (sp != std::string::npos) {
      const std::size_t s = cmd.find_first_not_of(' ', sp);
      if (s == std::string::npos) { break; }
      const std::size_t e = cmd.find(' ', s);
      const std::string word = cmd.substr(s, (e == std::string::npos) ? e : e - s);
      if (word.find('x') != std::string::npos) { ParseGeometry(word, opt.width, opt.height); }
      sp = e;
    }
    return Open(opt);
  }
  // A viewer that failed to open once will fail the same way for every later command in the
  // macro. Reporting the reason once and refusing the rest is more useful than eleven copies.
  if (g_open_failed) { return false; }
  // Before opening anything: the commands that need no window are answered here. Otherwise
  // `/vis/verbose confirmations` ahead of `/vis/open` opened the viewer at the default size
  // and swallowed the geometry string of the real /vis/open that followed.
  if (IsNoEffectVisCommand(cmd)) { return true; }
  if (!g_open && !Open(g_opt)) { return false; }
  return HandleVisCommand(cmd);
}

}  // namespace g4gpu::vis::viewer
