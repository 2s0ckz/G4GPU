// A minimal immediate-mode UI, drawn on the CPU into the same RGBA buffer the CUDA renderer
// fills.
//
// Why not a toolkit: the viewer already stages its frame through a host buffer on its way to a
// GL texture, so drawing widgets into that buffer costs one pass over a few thousand pixels
// and adds no dependency. Qt or ImGui would each be larger than the rest of this project.
//
// Why immediate mode: the widget state that matters - which control the mouse is over, which
// has focus, what is in a text field - is a handful of integers, and expressing the panel as
// straight-line code in the draw function keeps the layout and the behavior in one place.
// There is no widget tree to keep in sync with the model.
//
// Text comes from GDI. A fixed-pitch font is rendered once into a coverage atlas at startup,
// which avoids hand-typing glyph bitmaps and gives properly hinted text at any size the system
// has. gdi32 is already linked for the window itself.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace g4gpu::ui {

// ---------------------------------------------------------------- color

/// 0xAABBGGRR, matching the viewer's framebuffer layout.
using Color = unsigned int;

constexpr Color rgb(int r, int g, int b) {
  return 0xFF000000u | (static_cast<unsigned>(b & 255) << 16)
         | (static_cast<unsigned>(g & 255) << 8) | static_cast<unsigned>(r & 255);
}
constexpr Color rgba(int r, int g, int b, int a) {
  return (static_cast<unsigned>(a & 255) << 24) | (static_cast<unsigned>(b & 255) << 16)
         | (static_cast<unsigned>(g & 255) << 8) | static_cast<unsigned>(r & 255);
}

/// The palette. Dark, because the render behind it is dark and a light panel would flare.
namespace theme {
constexpr Color kPanel = rgb(28, 30, 34);
constexpr Color kPanelAlt = rgb(36, 39, 44);
constexpr Color kBorder = rgb(58, 62, 70);
constexpr Color kText = rgb(222, 226, 232);
constexpr Color kTextDim = rgb(140, 146, 156);
constexpr Color kAccent = rgb(86, 156, 214);
constexpr Color kAccentHot = rgb(120, 184, 236);
constexpr Color kOk = rgb(106, 176, 116);
constexpr Color kWarn = rgb(214, 158, 84);
constexpr Color kError = rgb(212, 100, 100);
constexpr Color kField = rgb(20, 22, 25);
}  // namespace theme

/// Blends `src` over `dst` using src's alpha.
inline Color blend(Color dst, Color src) {
  const unsigned a = (src >> 24) & 255;
  if (a == 255) { return src; }
  if (a == 0) { return dst; }
  const unsigned ia = 255 - a;
  const unsigned r = (((src >> 0) & 255) * a + ((dst >> 0) & 255) * ia) / 255;
  const unsigned g = (((src >> 8) & 255) * a + ((dst >> 8) & 255) * ia) / 255;
  const unsigned b = (((src >> 16) & 255) * a + ((dst >> 16) & 255) * ia) / 255;
  return 0xFF000000u | (b << 16) | (g << 8) | r;
}

// ---------------------------------------------------------------- font atlas

/// ASCII 32..126 rendered into a coverage atlas by GDI.
struct Font {
  int glyph_w = 0, glyph_h = 0;
  int ascent = 0;
  std::vector<unsigned char> coverage;  ///< glyph_w * glyph_h per character, 96 characters
  bool ready = false;

  static constexpr int kFirst = 32;
  static constexpr int kCount = 95;

  /// Builds the atlas. `face` may be any installed family; a fixed-pitch one keeps the
  /// per-character advance constant, which the layout code relies on.
  bool Build(const char* face, int pixel_height, bool bold = false) {
    HDC screen = GetDC(nullptr);
    HDC dc = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);
    if (dc == nullptr) { return false; }

    HFONT font = CreateFontA(-pixel_height, 0, 0, 0, bold ? FW_BOLD : FW_NORMAL, FALSE, FALSE,
                             FALSE, DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
                             CLEARTYPE_QUALITY, FIXED_PITCH | FF_MODERN, face);
    if (font == nullptr) {
      DeleteDC(dc);
      return false;
    }
    HGDIOBJ old_font = SelectObject(dc, font);

    TEXTMETRICA tm{};
    GetTextMetricsA(dc, &tm);
    glyph_w = tm.tmAveCharWidth;
    glyph_h = tm.tmHeight;
    ascent = tm.tmAscent;
    if (glyph_w <= 0 || glyph_h <= 0) {
      SelectObject(dc, old_font);
      DeleteObject(font);
      DeleteDC(dc);
      return false;
    }

    // One row of all glyphs, drawn white on black, then read back as coverage.
    const int atlas_w = glyph_w * kCount;
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = atlas_w;
    bi.bmiHeader.biHeight = -glyph_h;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bmp == nullptr) {
      SelectObject(dc, old_font);
      DeleteObject(font);
      DeleteDC(dc);
      return false;
    }
    HGDIOBJ old_bmp = SelectObject(dc, bmp);
    RECT all{0, 0, atlas_w, glyph_h};
    FillRect(dc, &all, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 255));
    for (int i = 0; i < kCount; ++i) {
      const char ch = static_cast<char>(kFirst + i);
      TextOutA(dc, i * glyph_w, 0, &ch, 1);
    }
    GdiFlush();

    coverage.assign(static_cast<size_t>(kCount) * glyph_w * glyph_h, 0);
    const unsigned* src = static_cast<const unsigned*>(bits);
    for (int i = 0; i < kCount; ++i) {
      for (int y = 0; y < glyph_h; ++y) {
        for (int x = 0; x < glyph_w; ++x) {
          const unsigned p = src[y * atlas_w + i * glyph_w + x];
          // ClearType writes different coverage per channel; the maximum is the closest thing
          // to a single-channel alpha and avoids the color fringing a naive average leaves.
          const unsigned r = (p >> 16) & 255, g = (p >> 8) & 255, b = p & 255;
          const unsigned v = (r > g) ? (r > b ? r : b) : (g > b ? g : b);
          coverage[(static_cast<size_t>(i) * glyph_h + y) * glyph_w + x] =
              static_cast<unsigned char>(v);
        }
      }
    }

    SelectObject(dc, old_bmp);
    DeleteObject(bmp);
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
    ready = true;
    return true;
  }

  int TextWidth(const std::string& s) const { return static_cast<int>(s.size()) * glyph_w; }
};

// ---------------------------------------------------------------- canvas

struct Rect {
  int x = 0, y = 0, w = 0, h = 0;
  bool Contains(int px, int py) const {
    return px >= x && px < x + w && py >= y && py < y + h;
  }
  Rect Inset(int d) const { return {x + d, y + d, w - 2 * d, h - 2 * d}; }
};

/// The overlap of two rectangles, empty when they miss.
///
/// Clipping is nested - a text field inside a scrolled panel inside a pop-up - and each level
/// has to *narrow* the clip it was handed rather than replace it, or the innermost widget
/// would draw outside the panel that contains it.
inline Rect ClipTo(const Rect& outer, const Rect& inner) {
  const int x0 = (inner.x > outer.x) ? inner.x : outer.x;
  const int y0 = (inner.y > outer.y) ? inner.y : outer.y;
  const int x1 = ((inner.x + inner.w) < (outer.x + outer.w)) ? (inner.x + inner.w)
                                                             : (outer.x + outer.w);
  const int y1 = ((inner.y + inner.h) < (outer.y + outer.h)) ? (inner.y + inner.h)
                                                             : (outer.y + outer.h);
  return Rect{x0, y0, (x1 > x0) ? x1 - x0 : 0, (y1 > y0) ? y1 - y0 : 0};
}

struct Canvas {
  unsigned* pixels = nullptr;
  int width = 0, height = 0;
  const Font* font = nullptr;
  /// Clip rectangle; drawing outside it is discarded. Set by BeginClip.
  Rect clip{0, 0, 0, 0};

  void Reset(unsigned* px, int w, int h, const Font* f) {
    pixels = px;
    width = w;
    height = h;
    font = f;
    clip = {0, 0, w, h};
  }

  void Put(int x, int y, Color c) {
    if (x < clip.x || y < clip.y || x >= clip.x + clip.w || y >= clip.y + clip.h) { return; }
    if (x < 0 || y < 0 || x >= width || y >= height) { return; }
    unsigned& d = pixels[static_cast<size_t>(y) * width + x];
    d = blend(d, c);
  }

  void FillRect(const Rect& r, Color c) {
    for (int y = r.y; y < r.y + r.h; ++y) {
      for (int x = r.x; x < r.x + r.w; ++x) { Put(x, y, c); }
    }
  }

  void StrokeRect(const Rect& r, Color c) {
    for (int x = r.x; x < r.x + r.w; ++x) {
      Put(x, r.y, c);
      Put(x, r.y + r.h - 1, c);
    }
    for (int y = r.y; y < r.y + r.h; ++y) {
      Put(r.x, y, c);
      Put(r.x + r.w - 1, y, c);
    }
  }

  /// A filled triangle, for the one shape this GUI needs that is not a rectangle: an arrowhead.
  ///
  /// Scanline between the two edges that span each row, every pixel through Put so the clip
  /// and the alpha blend are the same as everything else. Small enough to be worth having in
  /// preference to approximating an arrow out of shrinking rectangles, which is what a
  /// rectangle-only canvas pushes you towards and which does not read as an arrow at all.
  void FillTriangle(int x0, int y0, int x1, int y1, int x2, int y2, Color c) {
    // Sort the three by y, so the sweep is top to bottom with one apex above and one below.
    if (y1 < y0) { int t = x0; x0 = x1; x1 = t; t = y0; y0 = y1; y1 = t; }
    if (y2 < y0) { int t = x0; x0 = x2; x2 = t; t = y0; y0 = y2; y2 = t; }
    if (y2 < y1) { int t = x1; x1 = x2; x2 = t; t = y1; y1 = y2; y2 = t; }
    if (y2 == y0) {
      // Degenerate: a horizontal line. Drawing it is closer to the intent than drawing
      // nothing, and it is what an arrow seen exactly edge-on should look like.
      int lo = (x0 < x1) ? x0 : x1;
      int hi = (x0 > x1) ? x0 : x1;
      if (x2 < lo) { lo = x2; }
      if (x2 > hi) { hi = x2; }
      HLine(lo, hi + 1, y0, c);
      return;
    }
    for (int y = y0; y <= y2; ++y) {
      // The long edge from the top vertex to the bottom one spans every row; the other side
      // is whichever of the two short edges this row falls in.
      const float t_long = static_cast<float>(y - y0) / static_cast<float>(y2 - y0);
      const float xa = x0 + (x2 - x0) * t_long;
      float xb;
      if (y < y1) {
        xb = (y1 == y0) ? static_cast<float>(x1)
                        : x0 + (x1 - x0) * (static_cast<float>(y - y0)
                                            / static_cast<float>(y1 - y0));
      } else {
        xb = (y2 == y1) ? static_cast<float>(x1)
                        : x1 + (x2 - x1) * (static_cast<float>(y - y1)
                                            / static_cast<float>(y2 - y1));
      }
      int lo = static_cast<int>(xa < xb ? xa : xb);
      int hi = static_cast<int>(xa > xb ? xa : xb);
      HLine(lo, hi + 1, y, c);
    }
  }

  void HLine(int x0, int x1, int y, Color c) {
    for (int x = x0; x < x1; ++x) { Put(x, y, c); }
  }
  void VLine(int x, int y0, int y1, Color c) {
    for (int y = y0; y < y1; ++y) { Put(x, y, c); }
  }

  /// Draws @p s, ellipsized to fit @p max_w pixels. Returns the advance width.
  ///
  /// Clipping alone is not enough: a label cut mid-glyph looks like a rendering fault, and a
  /// label cut at the container edge with nothing to say so looks like the whole label. An
  /// ellipsis says "there is more" in the one place the reader is looking.
  int TextFit(int x, int y, const std::string& s, Color c, int max_w) {
    if (font == nullptr || max_w <= 0) { return 0; }
    if (TextWidth(s) <= max_w) { return Text(x, y, s, c); }
    const int ell = TextWidth("...");
    if (max_w <= ell) { return Text(x, y, "...", c); }
    const int fit = (max_w - ell) / (font->glyph_w > 0 ? font->glyph_w : 1);
    if (fit <= 0) { return Text(x, y, "...", c); }
    return Text(x, y, s.substr(0, static_cast<std::size_t>(fit)) + "...", c);
  }

  int TextWidth(const std::string& s) const {
    return (font != nullptr) ? font->TextWidth(s) : 0;
  }

  /// The glyph box, never zero.
  ///
  /// Callers divide by the width to turn a pixel offset into a character index, and a font
  /// that failed to load would make that a division by zero at the first click in a text
  /// field. 8x14 is close to the default face, so a fallback layout is still usable.
  int GlyphW() const { return (font != nullptr && font->glyph_w > 0) ? font->glyph_w : 8; }
  int GlyphH() const { return (font != nullptr && font->glyph_h > 0) ? font->glyph_h : 14; }

  /// Draws @p s wrapped inside @p r, breaking at spaces. Returns the height used.
  ///
  /// For prose - the explanations in the pop-ups - where the alternative is a line that runs
  /// off the panel or gets cut. Breaking at a space rather than mid-word because these are
  /// sentences.
  int TextWrapped(const Rect& r, const std::string& s, Color c) {
    if (font == nullptr || font->glyph_w <= 0) { return 0; }
    const int cols = r.w / font->glyph_w;
    if (cols < 4) { return 0; }
    int y = r.y;
    std::size_t i = 0;
    while (i < s.size()) {
      std::size_t take = s.size() - i;
      if (take > static_cast<std::size_t>(cols)) {
        take = static_cast<std::size_t>(cols);
        // Back up to the last space, unless the word is longer than the line.
        std::size_t brk = take;
        while (brk > 0 && s[i + brk] != ' ') { --brk; }
        if (brk > 0) { take = brk; }
      }
      Text(r.x, y, s.substr(i, take), c);
      y += font->glyph_h;
      i += take;
      while (i < s.size() && s[i] == ' ') { ++i; }
      if (y + font->glyph_h > r.y + r.h) { break; }
    }
    return y - r.y;
  }

  /// Draws @p s with its top-left at (x, y). Returns the advance width.
  int Text(int x, int y, const std::string& s, Color c) {
    if (font == nullptr || !font->ready) { return 0; }
    const int gw = font->glyph_w, gh = font->glyph_h;
    int pen = x;
    for (char raw : s) {
      const int ch = static_cast<unsigned char>(raw);
      if (ch == '\n') { continue; }
      const int idx = ch - Font::kFirst;
      if (idx < 0 || idx >= Font::kCount) {
        pen += gw;
        continue;
      }
      const unsigned char* g = &font->coverage[(static_cast<size_t>(idx) * gh) * gw];
      for (int gy = 0; gy < gh; ++gy) {
        for (int gx = 0; gx < gw; ++gx) {
          const unsigned char cov = g[gy * gw + gx];
          if (cov == 0) { continue; }
          const unsigned a = ((c >> 24) & 255) * cov / 255;
          Put(pen + gx, y + gy, (c & 0x00FFFFFFu) | (a << 24));
        }
      }
      pen += gw;
    }
    return pen - x;
  }

  int TextRight(int right_x, int y, const std::string& s, Color c) {
    if (font == nullptr) { return 0; }
    return Text(right_x - font->TextWidth(s), y, s, c);
  }

  /// Centres @p s in @p r, ellipsized if it does not fit.
  ///
  /// The ellipsis is not decoration. Centring a label wider than its box spills it out of
  /// *both* ends, so a button whose caption outgrew it drew over whatever was beside it and
  /// over its own border - which is how "Reset to defaults" came to straddle the edge of the
  /// dialog it was in. Fitting it here fixes every such button at once; sizing each one by
  /// hand fixes the ones somebody noticed.
  int TextCentred(const Rect& r, const std::string& s, Color c) {
    if (font == nullptr) { return 0; }
    // The threshold is the *full* width, not a padded one. A label that reached the box's
    // edges was always drawn that way and reads fine; trimming those as well turned
    // "Save project" into "Save pro..." across the whole bottom panel. Only what genuinely
    // overflows is trimmed, and then to two pixels inside the border.
    const int ty = r.y + (r.h - font->glyph_h) / 2;
    if (font->TextWidth(s) <= r.w) {
      return Text(r.x + (r.w - font->TextWidth(s)) / 2, ty, s, c);
    }
    return TextFit(r.x + 2, ty, s, c, r.w - 4);
  }
};

// ---------------------------------------------------------------- input and state

/// Everything the window procedure collects between frames.
struct Input {
  int mouse_x = 0, mouse_y = 0;
  int wheel = 0;                ///< accumulated notches, consumed each frame
  bool left_down = false, right_down = false, middle_down = false;
  bool left_pressed = false, left_released = false;  ///< edges, consumed each frame
  bool left_dbl = false;        ///< the second click of a double-click, also a left_pressed
  bool right_pressed = false, right_released = false;
  std::string typed;            ///< printable characters since the last frame
  std::vector<int> keys;        ///< virtual key codes pressed since the last frame
  bool ctrl = false, shift = false;

  /// Called at the end of each frame: edges and queues are per-frame.
  void EndFrame() {
    wheel = 0;
    left_pressed = left_released = false;
    left_dbl = false;
    right_pressed = right_released = false;
    typed.clear();
    keys.clear();
  }
  bool KeyPressed(int vk) const {
    for (int k : keys) {
      if (k == vk) { return true; }
    }
    return false;
  }
};

/// Widget identity and interaction state. Ids are pointers-worth-of-int; using the address of
/// a stable object, or a hand-assigned constant, both work.
struct Context {
  Canvas canvas;
  Input* in = nullptr;
  int hot = 0;       ///< widget under the cursor
  int active = 0;    ///< widget being dragged or held
  int focus = 0;     ///< widget receiving typed characters
  int next_auto_id = 1000;

  // The focused text field's editing state. It lives here, not in the field, because exactly
  // one field is focused: a per-field copy would be state to keep in step with which field
  // that is, and the answer would always be "the focused one".
  int caret = 0;          ///< character index the caret sits before; -1 means "select all"
  int sel_anchor = 0;     ///< the other end of the selection; == caret means no selection
  int text_dragging = 0;  ///< field whose selection the mouse is currently extending
  int text_scroll = 0;    ///< pixels the focused field's text is scrolled left

  /// Which stack of things is being drawn: panels behind, then the menu bar, then pop-ups.
  ///
  /// It exists for one reason - an open dropdown has to be painted after everything at its
  /// OWN layer and before anything in front of it, and the widget that opened it is long
  /// finished by then. A single "draw the open list after the panels" was right for a
  /// dropdown in a panel and wrong for one inside a pop-up: the pop-up painted straight over
  /// its own open list. See DrawOpenSelect.
  ///
  /// Set with LayerScope around each stack rather than assigned by hand, because the failure
  /// from forgetting to put it back is a dropdown that renders behind something once every
  /// few frames, which is a miserable thing to chase.
  enum Layer : int { kLayerPanel = 0, kLayerMenu = 1, kLayerPopup = 2 };
  int layer = kLayerPanel;

  // The open dropdown, if any. See Select() and DrawOpenSelect() at the foot of this file:
  // the list has to be painted after everything at its own layer, so what is open has to
  // outlive the widget that declared it.
  int open_select = 0;                        ///< id of the Select whose list is open, or 0
  int select_layer = kLayerPanel;             ///< the layer that Select was declared at
  Rect select_rect{};                         ///< where that Select's button was
  const char* const* select_opts = nullptr;   ///< its options; static literals, so no copy
  int select_count = 0;
  int select_picked = -1;                     ///< row clicked last frame, consumed by Select

  /// Focuses a text field from code, with its whole value selected.
  ///
  /// For the case where the *caller* decides a field should be editing - starting a rename
  /// from a second click on a row, say - where there is no click position to place a caret
  /// from. Selecting everything is right there: the first thing typed replaces the old name,
  /// and an arrow key or a click still gets you to editing it in place.
  ///
  /// -1 is a "select all when you are next drawn" sentinel rather than the length, because
  /// the caller would have to pass the length and the field already knows it.
  void FocusText(int id) {
    focus = id;
    caret = -1;
    sel_anchor = -1;
    text_scroll = 0;
  }

  void Begin(unsigned* px, int w, int h, const Font* f, Input* input) {
    canvas.Reset(px, w, h, f);
    in = input;
    hot = 0;
  }
  void End() {
    if (in != nullptr && in->left_released) {
      active = 0;
      // Also here, and not only in the field: a drag released over a panel that has since
      // scrolled the field out of view would leave it dragging forever, because the code
      // that clears it is the code that is no longer drawn.
      text_dragging = 0;
    }
  }

  bool Hovering(const Rect& r) const {
    return in != nullptr && r.Contains(in->mouse_x, in->mouse_y);
  }
};

// ---------------------------------------------------------------- scrolling

/// A scrollable region.
///
/// Immediate-mode scrolling without a retained layout: the caller draws at `y - offset`, the
/// canvas clips to the viewport, and the content height measured this frame sets the scroll
/// range for the next one. One frame of lag on the range is invisible and costs nothing;
/// measuring twice per frame to avoid it would double the panel code.
///
/// The offset is clamped every frame, so a list that shrinks - a delete - pulls the view back
/// up instead of leaving it scrolled past the end.
/// Width of a scroll area's bar, and of the strip along its right edge that grabs it.
///
/// It sits in the panel's own right margin rather than taking width from the content: a bar
/// that reserved width would change the content's layout, which changes the content height,
/// which decides whether there is a bar at all - an oscillation between two layouts on
/// alternate frames. Every section here already leaves eight pixels at the right, so the bar
/// has somewhere to be.
inline constexpr int kScrollBarW = 9;

struct ScrollArea {
  int offset = 0;      ///< pixels scrolled down
  int content_h = 0;   ///< height the content wanted last frame
  int view_h = 0;

  /// The bar's track, and the thumb inside it. Empty when there is nothing to scroll.
  ///
  /// Both Begin (which drags it) and End (which draws it) need this, and from the SAME
  /// numbers: a thumb drawn anywhere other than where the drag picks it up is a control that
  /// jumps out from under the cursor.
  void BarRects(const Rect& r, Rect& track, Rect& thumb) const {
    track = Rect{0, 0, 0, 0};
    thumb = track;
    if (content_h <= r.h || r.h <= 0) { return; }
    track = Rect{r.x + r.w - kScrollBarW, r.y, kScrollBarW, r.h};
    int th = (r.h * r.h) / content_h;
    if (th < 16) { th = 16; }            // still grabbable at the far end of a long list
    if (th > r.h) { th = r.h; }
    const int max_off = content_h - r.h;
    const int ty = r.y + ((r.h - th) * offset) / (max_off > 0 ? max_off : 1);
    thumb = Rect{track.x + 1, ty, kScrollBarW - 2, th};
  }

  /// Clips to @p r, handles the wheel and a drag of the bar, and returns the y to start
  /// drawing at.
  ///
  /// The bar is handled HERE, before the content is drawn, and not in End: a click is claimed
  /// by the first widget that tests for it, and by End every row in the list has already had
  /// its turn. Both work from the previous frame's content height, which is the only one
  /// there is until the content has been drawn - and which is what the wheel clamp below has
  /// always used.
  int Begin(Context& ctx, const Rect& r, int step, int id = 0) {
    view_h = r.h;
    const int max_off = (content_h > r.h) ? (content_h - r.h) : 0;
    if (ctx.Hovering(r) && ctx.in != nullptr && ctx.in->wheel != 0) {
      offset -= ctx.in->wheel * step;
      ctx.in->wheel = 0;  // consumed: the camera must not also zoom
    }
    id_ = id;
    if (id != 0 && ctx.in != nullptr && max_off > 0) {
      Rect track, thumb;
      BarRects(r, track, thumb);
      const bool on_track = track.Contains(ctx.in->mouse_x, ctx.in->mouse_y);
      if (on_track) { ctx.hot = id; }
      if (on_track && ctx.in->left_pressed && ctx.active == 0) {
        ctx.active = id;
        // Grabbing the thumb keeps the point under the cursor; clicking the empty track
        // jumps the thumb to the cursor and drags from its middle, which is what a click on
        // a track that is mostly empty is asking for.
        grab_dy_ = thumb.Contains(ctx.in->mouse_x, ctx.in->mouse_y)
                       ? (ctx.in->mouse_y - thumb.y)
                       : (thumb.h / 2);
        ctx.in->left_pressed = false;   // consumed, so no row underneath also takes it
      }
      if (ctx.active == id) {
        if (ctx.in->left_down) {
          Rect t2, th2;
          BarRects(r, t2, th2);
          const int span = r.h - th2.h;
          const int want = ctx.in->mouse_y - grab_dy_ - r.y;
          offset = (span > 0) ? static_cast<int>(static_cast<long long>(want) * max_off / span)
                              : 0;
        } else {
          ctx.active = 0;
        }
      }
    }
    if (offset > max_off) { offset = max_off; }
    if (offset < 0) { offset = 0; }
    saved_clip_ = ctx.canvas.clip;
    // Intersect rather than replace: a scroll area inside a popup must not draw outside it.
    Rect cl = r;
    const int x0 = (cl.x > saved_clip_.x) ? cl.x : saved_clip_.x;
    const int y0 = (cl.y > saved_clip_.y) ? cl.y : saved_clip_.y;
    const int x1 = ((cl.x + cl.w) < (saved_clip_.x + saved_clip_.w)) ? (cl.x + cl.w)
                                                                     : (saved_clip_.x + saved_clip_.w);
    const int y1 = ((cl.y + cl.h) < (saved_clip_.y + saved_clip_.h)) ? (cl.y + cl.h)
                                                                     : (saved_clip_.y + saved_clip_.h);
    ctx.canvas.clip = Rect{x0, y0, (x1 > x0) ? x1 - x0 : 0, (y1 > y0) ? y1 - y0 : 0};
    top_ = r.y;
    return r.y - offset;
  }

  /// Restores the clip and records the content height. @p y_end is where the caller stopped.
  void End(Context& ctx, const Rect& r, int y_end) {
    content_h = y_end + offset - top_;
    ctx.canvas.clip = saved_clip_;
    // Drawn from the height just measured, so the bar matches the content on the frame it is
    // drawn on. Begin dragged it from the previous frame's height, which is a frame stale in
    // the middle of a drag and invisible at any scroll speed a hand produces.
    Rect track, thumb;
    BarRects(r, track, thumb);
    if (track.h > 0) {
      const bool lit = (ctx.active == id_ && id_ != 0)
                       || (ctx.in != nullptr
                           && track.Contains(ctx.in->mouse_x, ctx.in->mouse_y));
      ctx.canvas.FillRect(track, theme::kField);
      ctx.canvas.FillRect(thumb, lit ? theme::kAccent : theme::kBorder);
    }
  }

 private:
  Rect saved_clip_{};
  int top_ = 0;
  int id_ = 0;
  int grab_dy_ = 0;   ///< where in the thumb the drag was picked up
};

// ---------------------------------------------------------------- widgets

inline void Panel(Context& ctx, const Rect& r, Color fill = theme::kPanel) {
  ctx.canvas.FillRect(r, fill);
  ctx.canvas.StrokeRect(r, theme::kBorder);
}

inline void Label(Context& ctx, int x, int y, const std::string& s,
                  Color c = theme::kText) {
  ctx.canvas.Text(x, y, s, c);
}

/// A section heading with a rule under it. Returns the y below the rule.
/// Draws a section's title where it will not scroll, and returns the rect left underneath.
///
/// The title used to be the first thing inside the scroll area, so scrolling a long list took
/// the word SOLIDS off the top of the panel with it and left a column of rows belonging to
/// nothing. A heading that scrolls away is a heading that stops being one exactly when the
/// list is long enough to need it.
inline Rect FrozenTitle(Context& ctx, const Rect& area, const std::string& title) {
  ctx.canvas.Text(area.x + 8, area.y, title, theme::kTextDim);
  const int line_y = area.y + ctx.canvas.font->glyph_h + 3;
  ctx.canvas.HLine(area.x + 8, area.x + area.w - 8, line_y, theme::kBorder);
  const int used = line_y + 6 - area.y;
  return Rect{area.x, line_y + 6, area.w, (area.h > used) ? area.h - used : 0};
}

/// Splits @p area into a list above and the selected item's form below.
///
/// The form is not part of the list: scrolling down a hundred solids to see the hundredth
/// used to scroll its fields off the bottom, so the two things you need at once could not
/// both be on screen. They get separate scroll areas.
///
/// THE LIST TAKES WHAT IT NEEDS AND THE FORM TAKES THE REST, so the form always begins
/// immediately under the last row rather than at a fixed height - a fixed split leaves a band
/// of nothing between a two-source list and its form. The list is capped so a long one cannot
/// crowd the form out: the cap leaves the form whatever it asked for, or half the section,
/// whichever is less.
///
/// Half, and not a count of rows: the first version of this left the list a floor of three
/// rows, and a solid form is a dozen fields and six buttons, so an eleven-solid list came out
/// three rows tall. A floor that is a SHARE of the space scales with the panel instead of
/// with a row height guessed here - the same lesson the section heights taught.
///
/// Neither height can feed back on the other: a form is a fixed set of rows for a given shape
/// and a list a fixed set of rows for a given model, so neither reflows when its space
/// changes. That is what makes it safe to size them from what they asked for last frame.
///
/// @p want_list, @p want_form  content heights from the previous frame.
inline void SplitListForm(const Rect& area, int want_list, int want_form, Rect& list,
                          Rect& form) {
  int keep = (want_form > 0) ? want_form : 0;   // what the form needs
  const int share = area.h / 2;
  if (keep > share) { keep = share; }           // but never more than half
  int lh = (want_list > 0) ? want_list : 0;
  if (lh > area.h - keep) { lh = area.h - keep; }
  if (lh < 0) { lh = 0; }
  list = Rect{area.x, area.y, area.w, lh};
  form = Rect{area.x, area.y + lh, area.w, area.h - lh};
}

inline int SectionHeader(Context& ctx, const Rect& area, int y, const std::string& title) {
  ctx.canvas.Text(area.x + 8, y, title, theme::kTextDim);
  const int line_y = y + ctx.canvas.font->glyph_h + 3;
  ctx.canvas.HLine(area.x + 8, area.x + area.w - 8, line_y, theme::kBorder);
  return line_y + 6;
}

/// A push button. Returns true on the frame it is released over.
inline bool Button(Context& ctx, int id, const Rect& r, const std::string& label,
                   bool enabled = true) {
  const bool hover = enabled && ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  if (hover && ctx.in->left_pressed) {
    ctx.active = id;
    ctx.focus = 0;
  }
  const bool clicked = enabled && ctx.active == id && hover && ctx.in->left_released;

  Color fill = theme::kPanelAlt;
  Color text = theme::kText;
  if (!enabled) {
    text = theme::kTextDim;
  } else if (ctx.active == id && hover) {
    fill = theme::kAccent;
  } else if (hover) {
    fill = theme::kAccentHot;
    text = rgb(16, 18, 20);
  }
  ctx.canvas.FillRect(r, fill);
  ctx.canvas.StrokeRect(r, enabled ? theme::kBorder : theme::kPanelAlt);
  ctx.canvas.TextCentred(r, label, text);
  return clicked;
}

/// A toggle. `value` is modified in place; returns true when it changed.
inline bool Checkbox(Context& ctx, int id, const Rect& r, const std::string& label,
                     bool& value) {
  const int box = ctx.canvas.font->glyph_h;
  const Rect box_r{r.x, r.y + (r.h - box) / 2, box, box};
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  bool changed = false;
  if (hover && ctx.in->left_pressed) {
    value = !value;
    changed = true;
    ctx.focus = 0;
  }
  ctx.canvas.FillRect(box_r, theme::kField);
  ctx.canvas.StrokeRect(box_r, hover ? theme::kAccentHot : theme::kBorder);
  if (value) { ctx.canvas.FillRect(box_r.Inset(3), theme::kAccent); }
  ctx.canvas.Text(r.x + box + 8, r.y + (r.h - ctx.canvas.font->glyph_h) / 2, label,
                  theme::kText);
  return changed;
}

/// A text buffer that belongs to one edited thing at a time.
///
/// An immediate-mode text field needs somewhere to keep the partially typed string, and that
/// somewhere has to be per *field* - not shared. The builder had one `name_edit` string and an
/// integer tag saying which field owned it, reset with
///
///     if (tag != 10000 + selected_element) { text = element.symbol; tag = ...; }
///
/// at four different sites. Every one of them ran every frame, so with an element and a
/// material both selected the two blocks overwrote each other's text in turn and every
/// character typed was gone by the next frame. That is why typing into the builder stopped
/// working as soon as anything was added: before that, nothing was selected and only one site
/// ran.
///
/// One of these per field site fixes it by construction: the key identifies what is being
/// edited, and a buffer only ever holds one field's text.
struct EditBuffer {
  int key = -2;
  std::string text;

  /// The buffer for thing @p k, seeded from @p current the first time it is asked for.
  std::string& For(int k, const std::string& current) {
    if (key != k) {
      key = k;
      text = current;
    }
    return text;
  }
  /// Forget what is held, so the next For() re-seeds. Call after writing the text back into
  /// the document from somewhere other than the field itself.
  void Reset() { key = -2; }
};

/// An editable text field, with a caret and a selection.
///
/// The caret, the selection anchor and the horizontal scroll live in the Context rather than
/// in the field, because exactly one field has focus at a time: a per-field copy would be
/// three more things to keep in step with which field that is, and the answer would always be
/// "the focused one". Losing focus is what resets them.
///
/// What works: click to place the caret, drag to select, shift with the arrows and Home/End,
/// double-click to select all, Ctrl+A, and Ctrl+C / Ctrl+X / Ctrl+V through the Windows
/// clipboard. Typing, Backspace or Delete with a selection replaces it. Double-click selects
/// the whole field rather than a word - for a field holding `G4_WATER` or `1.35 g/cm3` that is
/// what is wanted, and a word-break rule would have to decide what `/` and `.` are.
///
/// The text scrolls horizontally to keep the caret in view, so a value longer than the box is
/// editable at its far end. That is why the focused field does not go through TextFit: an
/// ellipsis would make the character index stop matching the pixel position, and every click
/// after it would land somewhere else.
///
/// Returns true when Enter is pressed while it has focus.
inline bool TextField(Context& ctx, int id, const Rect& r, std::string& text,
                      const std::string& placeholder = "", int max_len = 64) {
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  const int gw = ctx.canvas.GlyphW();
  constexpr int kPad = 6;
  const int len = static_cast<int>(text.size());

  // The character index under an x coordinate: rounded to the nearest gap, so clicking the
  // left half of a character puts the caret before it. Fixed-width font, hence exact.
  auto index_at = [&](int px) {
    int i = (px - (r.x + kPad) + ctx.text_scroll + gw / 2) / gw;
    if (i < 0) { i = 0; }
    if (i > static_cast<int>(text.size())) { i = static_cast<int>(text.size()); }
    return i;
  };

  if (hover && ctx.in != nullptr && ctx.in->left_pressed) {
    if (ctx.focus != id) {
      ctx.focus = id;
      ctx.text_scroll = 0;
    }
    if (ctx.in->left_dbl) {
      // Select all. A word-break rule would have to decide what `/` and `.` are, and for a
      // field holding `G4_WATER` or `1.35 g/cm3` the whole value is what a double-click is
      // reaching for anyway - it is how you replace one.
      ctx.sel_anchor = 0;
      ctx.caret = static_cast<int>(text.size());
    } else {
      ctx.caret = index_at(ctx.in->mouse_x);
      ctx.sel_anchor = ctx.caret;
      ctx.text_dragging = id;
    }
  }
  if (ctx.text_dragging == id && ctx.in != nullptr) {
    if (ctx.in->left_down) {
      // Dragging past either edge selects towards that end, which is what the pointer being
      // outside the box is asking for.
      ctx.caret = index_at(ctx.in->mouse_x);
    } else {
      ctx.text_dragging = 0;
    }
  }
  const bool focused = (ctx.focus == id);

  // Clamped every frame: the caller may have replaced the text behind the field's back - a
  // draft re-synced from the document does exactly that - and an index past the end would
  // then erase from nowhere.
  if (focused) {
    if (ctx.caret < 0) {
      // FocusText's sentinel: this is the first frame after code took focus, so select all.
      ctx.sel_anchor = 0;
      ctx.caret = len;
    }
    if (ctx.caret > len) { ctx.caret = len; }
    if (ctx.sel_anchor > len) { ctx.sel_anchor = len; }
    if (ctx.sel_anchor < 0) { ctx.sel_anchor = 0; }
  }

  const int lo = (ctx.caret < ctx.sel_anchor) ? ctx.caret : ctx.sel_anchor;
  const int hi = (ctx.caret < ctx.sel_anchor) ? ctx.sel_anchor : ctx.caret;
  const bool has_sel = focused && (lo != hi);

  bool submitted = false;
  if (focused && ctx.in != nullptr) {
    auto erase_sel = [&]() {
      if (ctx.caret == ctx.sel_anchor) { return; }
      const int a = (ctx.caret < ctx.sel_anchor) ? ctx.caret : ctx.sel_anchor;
      const int b = (ctx.caret < ctx.sel_anchor) ? ctx.sel_anchor : ctx.caret;
      text.erase(static_cast<std::size_t>(a), static_cast<std::size_t>(b - a));
      ctx.caret = a;
      ctx.sel_anchor = a;
    };
    auto insert = [&](char c) {
      erase_sel();
      if (static_cast<int>(text.size()) >= max_len) { return; }
      text.insert(text.begin() + ctx.caret, c);
      ++ctx.caret;
      ctx.sel_anchor = ctx.caret;
    };

    // Ctrl combinations are handled from the key queue, not the character queue: WM_CHAR
    // delivers Ctrl+A as 0x01 and Ctrl+C as 0x03, which the printable filter below would
    // drop silently - and Ctrl+V as 0x16, which it would drop *after* the paste, or before
    // it, depending on nothing in particular.
    if (ctx.in->ctrl) {
      if (ctx.in->KeyPressed('A')) {
        ctx.sel_anchor = 0;
        ctx.caret = static_cast<int>(text.size());
      }
      const bool copy = ctx.in->KeyPressed('C');
      const bool cut = ctx.in->KeyPressed('X');
      if ((copy || cut) && ctx.caret != ctx.sel_anchor) {
        const int a = (ctx.caret < ctx.sel_anchor) ? ctx.caret : ctx.sel_anchor;
        const int b = (ctx.caret < ctx.sel_anchor) ? ctx.sel_anchor : ctx.caret;
        const std::string s = text.substr(static_cast<std::size_t>(a),
                                          static_cast<std::size_t>(b - a));
        if (OpenClipboard(nullptr) != 0) {
          EmptyClipboard();
          HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, s.size() + 1);
          if (h != nullptr) {
            void* p = GlobalLock(h);
            if (p != nullptr) {
              std::memcpy(p, s.c_str(), s.size() + 1);
              GlobalUnlock(h);
              // SetClipboardData takes ownership on success; GlobalFree would double-free.
              if (SetClipboardData(CF_TEXT, h) == nullptr) { GlobalFree(h); }
            } else {
              GlobalFree(h);
            }
          }
          CloseClipboard();
        }
        if (cut) { erase_sel(); }
      }
      if (ctx.in->KeyPressed('V') && OpenClipboard(nullptr) != 0) {
        HANDLE h = GetClipboardData(CF_TEXT);
        if (h != nullptr) {
          const char* p = static_cast<const char*>(GlobalLock(h));
          if (p != nullptr) {
            erase_sel();
            // Newlines and tabs are dropped rather than pasted: a single-line field with a
            // newline in it is a value no parser here accepts, and pasting one line of a
            // multi-line clipboard is a guess about which line.
            for (; *p != 0; ++p) {
              if (*p >= 32 && *p < 127) { insert(*p); }
            }
            GlobalUnlock(h);
          }
        }
        CloseClipboard();
      }
    } else {
      for (char c : ctx.in->typed) {
        if (c == '\r' || c == '\n') {
          submitted = true;
        } else if (c == '\b') {
          if (ctx.caret != ctx.sel_anchor) {
            erase_sel();
          } else if (ctx.caret > 0) {
            text.erase(text.begin() + (ctx.caret - 1));
            --ctx.caret;
            ctx.sel_anchor = ctx.caret;
          }
        } else if (c >= 32 && c < 127) {
          insert(c);
        }
      }
    }

    // Caret movement. Shift extends by leaving the anchor where it is; without shift the
    // anchor follows, which collapses the selection.
    auto move_to = [&](int to) {
      if (to < 0) { to = 0; }
      if (to > static_cast<int>(text.size())) { to = static_cast<int>(text.size()); }
      ctx.caret = to;
      if (!ctx.in->shift) { ctx.sel_anchor = to; }
    };
    if (ctx.in->KeyPressed(VK_LEFT)) {
      // An unshifted left arrow with a selection goes to its start, not one left of the
      // caret: that is what every other text box does.
      move_to((!ctx.in->shift && ctx.caret != ctx.sel_anchor) ? lo : ctx.caret - 1);
    }
    if (ctx.in->KeyPressed(VK_RIGHT)) {
      move_to((!ctx.in->shift && ctx.caret != ctx.sel_anchor) ? hi : ctx.caret + 1);
    }
    if (ctx.in->KeyPressed(VK_HOME)) { move_to(0); }
    if (ctx.in->KeyPressed(VK_END)) { move_to(static_cast<int>(text.size())); }
    if (ctx.in->KeyPressed(VK_DELETE)) {
      if (ctx.caret != ctx.sel_anchor) {
        erase_sel();
      } else if (ctx.caret < static_cast<int>(text.size())) {
        text.erase(text.begin() + ctx.caret);
      }
    }
    if (ctx.in->KeyPressed(VK_ESCAPE)) { ctx.focus = 0; }
  }

  // ---- drawing

  ctx.canvas.FillRect(r, theme::kField);
  ctx.canvas.StrokeRect(r, focused ? theme::kAccent : (hover ? theme::kAccentHot
                                                             : theme::kBorder));
  const int ty = r.y + (r.h - ctx.canvas.GlyphH()) / 2;
  const int inner_w = r.w - 2 * kPad;
  if (text.empty() && !focused) {
    if (!placeholder.empty()) {
      ctx.canvas.TextFit(r.x + kPad, ty, placeholder, theme::kTextDim, inner_w);
    }
    return submitted;
  }
  if (!focused) {
    // Unfocused and too long: an ellipsis, since there is no caret to keep in view and a hard
    // cut in the middle of a word reads as a different value.
    ctx.canvas.TextFit(r.x + kPad, ty, text, theme::kText, inner_w);
    return submitted;
  }

  // Keep the caret inside the box. Only the focused field scrolls, and only one field is
  // focused, so this single offset is the whole story.
  const int caret_px = ctx.caret * gw;
  if (caret_px - ctx.text_scroll > inner_w - gw) { ctx.text_scroll = caret_px - inner_w + gw; }
  if (caret_px - ctx.text_scroll < 0) { ctx.text_scroll = caret_px; }
  const int text_px = static_cast<int>(text.size()) * gw;
  if (ctx.text_scroll > text_px - inner_w + gw) { ctx.text_scroll = text_px - inner_w + gw; }
  if (ctx.text_scroll < 0) { ctx.text_scroll = 0; }

  const Rect saved_clip = ctx.canvas.clip;
  ctx.canvas.clip = ClipTo(saved_clip, {r.x + kPad - 1, r.y + 1, inner_w + 2, r.h - 2});
  const int tx = r.x + kPad - ctx.text_scroll;
  if (has_sel) {
    ctx.canvas.FillRect({tx + lo * gw, ty - 1, (hi - lo) * gw, ctx.canvas.GlyphH() + 2},
                        rgba(58, 106, 168, 255));
  }
  ctx.canvas.Text(tx, ty, text, theme::kText);
  // A static caret: a blinking one needs a clock in the context and buys nothing here.
  ctx.canvas.VLine(tx + caret_px, ty, ty + ctx.canvas.GlyphH(), theme::kAccent);
  ctx.canvas.clip = saved_clip;
  return submitted;
}

/// A text field with completion from a candidate list.
///
/// The completion is shown as dim text after the caret and accepted with Tab or Enter, which
/// is the one interaction that works without a popup list stealing the click. `candidates` is
/// walked once per frame; with a few hundred NIST names that is nothing next to the blit.
///
/// Case-insensitive prefix matching, because a user typing a NIST name types `g4_wat` and the
/// table says `G4_WATER`.
///
/// @param[out] accepted set when the completion was taken, so the caller can commit
/// @return true when Enter was pressed
template <typename Candidates>
inline bool TextFieldAuto(Context& ctx, int id, const Rect& r, std::string& text,
                          const Candidates& candidates, const std::string& placeholder = "",
                          int max_len = 64, bool* accepted = nullptr) {
  const bool focused_before = (ctx.focus == id);
  // Tab is read before the field consumes the typed characters, so it completes rather than
  // inserting a tab.
  bool take = false;
  if (focused_before && ctx.in != nullptr) {
    for (char ch : ctx.in->typed) {
      if (ch == '\t') { take = true; }
    }
  }
  const bool submitted = TextField(ctx, id, r, text, placeholder, max_len);
  if (accepted != nullptr) { *accepted = false; }
  if (ctx.focus != id || text.empty()) { return submitted; }

  // The first candidate that has `text` as a prefix, ignoring case.
  const char* best = nullptr;
  for (const char* cand : candidates) {
    if (cand == nullptr) { continue; }
    std::size_t k = 0;
    for (; k < text.size(); ++k) {
      const char x = text[k], y = cand[k];
      if (y == '\0') { break; }
      const char lx = (x >= 'A' && x <= 'Z') ? static_cast<char>(x + 32) : x;
      const char ly = (y >= 'A' && y <= 'Z') ? static_cast<char>(y + 32) : y;
      if (lx != ly) { break; }
    }
    if (k == text.size()) {
      best = cand;
      break;
    }
  }
  if (best == nullptr) { return submitted; }

  const std::string full = best;
  if (take || (submitted && full.size() != text.size())) {
    text = full;
    // The caret goes to the end of what was accepted. Leaving it where it was would put it
    // in the middle of a name the user did not type, and the next keystroke would insert
    // there - the completion has to move the caret with it.
    ctx.caret = static_cast<int>(full.size());
    ctx.sel_anchor = ctx.caret;
    if (accepted != nullptr) { *accepted = true; }
    return submitted;
  }
  // The tail, drawn dim after what has been typed - but only with the caret at the end and
  // nothing selected, which is the only state where Tab means "take the rest of this name".
  // With a selection, or the caret mid-string, a ghost tail is describing an edit that Tab
  // would not make.
  if (full.size() > text.size() && ctx.caret == static_cast<int>(text.size())
      && ctx.sel_anchor == ctx.caret) {
    const int cx = r.x + 6 + ctx.canvas.TextWidth(text) - ctx.text_scroll;
    const int ty = r.y + (r.h - ctx.canvas.GlyphH()) / 2;
    const Rect saved = ctx.canvas.clip;
    ctx.canvas.clip = ClipTo(saved, {r.x + 5, r.y + 1, r.w - 10, r.h - 2});
    ctx.canvas.Text(cx + 1, ty, full.substr(text.size()), theme::kTextDim);
    ctx.canvas.clip = saved;
    ctx.canvas.TextRight(r.x + r.w - 4, ty, "tab", theme::kBorder);
  }
  return submitted;
}


/// A numeric field. Keeps its own text so that a partially typed number is not clobbered by
/// re-formatting the value every frame - the reason a naive "format the double back into the
/// box" field cannot be typed into.
struct NumberField {
  std::string text;
  double value = 0;
  bool Init(double v, const char* fmt = "%g") {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    text = buf;
    value = v;
    return true;
  }
  /// Returns true when the value changed.
  bool Commit() {
    const double v = std::atof(text.c_str());
    if (v != value) {
      value = v;
      return true;
    }
    return false;
  }
};

inline bool NumberInput(Context& ctx, int id, const Rect& r, NumberField& f,
                        const std::string& placeholder = "", bool enabled = true) {
  if (!enabled) {
    // Drawn, not omitted: the value still matters - it is what the field would say if it were
    // enabled - and a row that appears and disappears is harder to read than one that greys.
    // Nothing is hit-tested, so it cannot take focus or be typed into.
    ctx.canvas.FillRect(r, theme::kPanel);
    ctx.canvas.StrokeRect(r, theme::kBorder);
    ctx.canvas.TextFit(r.x + 6, r.y + (r.h - ctx.canvas.GlyphH()) / 2, f.text, theme::kTextDim,
                       r.w - 12);
    if (ctx.focus == id) { ctx.focus = 0; }
    return false;
  }
  const bool submitted = TextField(ctx, id, r, f.text, placeholder, 24);
  const bool lost_focus = (ctx.focus != id);
  if (submitted || lost_focus) { return f.Commit(); }
  return false;
}

/// A horizontal slider over [lo, hi]. Returns true while being dragged.
inline bool Slider(Context& ctx, int id, const Rect& r, double lo, double hi, double& value) {
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  if (hover && ctx.in->left_pressed) {
    ctx.active = id;
    ctx.focus = 0;
  }
  bool dragging = false;
  if (ctx.active == id && ctx.in->left_down) {
    const double f = static_cast<double>(ctx.in->mouse_x - r.x) / (r.w > 0 ? r.w : 1);
    value = lo + (hi - lo) * (f < 0 ? 0 : (f > 1 ? 1 : f));
    dragging = true;
  }
  const double f = (hi > lo) ? (value - lo) / (hi - lo) : 0.0;
  const int fill_w = static_cast<int>(f * r.w);
  ctx.canvas.FillRect(r, theme::kField);
  ctx.canvas.FillRect({r.x, r.y, fill_w, r.h}, theme::kAccent);
  ctx.canvas.StrokeRect(r, hover ? theme::kAccentHot : theme::kBorder);
  return dragging;
}

/// A selectable row, for the element / material / solid lists. Returns true when clicked.
/// Where ListRow draws its swatch, for a caller that wants the swatch to be clickable.
///
/// Here rather than in each caller because it was in each caller: the solids list computed
/// `{row.x + 6, ..., lh - 2, lh - 2}` by hand to make its swatch a hit target, and the voxel
/// class rows below it did not - so clicking a class's colour opened the material picker,
/// which is the bug this exists to stop recurring. One definition, three users.
/// A visibility toggle, drawn as an eye. Returns true on the frame it was clicked.
///
/// DRAWN, not typed. The font here is an atlas of the 95 printable ASCII characters (see
/// Font::kFirst), so U+1F441 and every other pictograph is not available to it at any size -
/// it would come out as whatever glyph 0x1F441 minus 32 happens to land on, which is nothing.
/// Two triangles and a pupil is the whole icon.
///
/// Hidden keeps the outline and loses the pupil, with a line through it: an empty column
/// reads as "this row has no control" rather than "this row is switched off", and the two
/// have to be told apart at a glance down a list of a hundred.
/// A family of units one quantity can be quoted in, and what one of each is in the base unit.
///
/// THE MODEL IS ALWAYS IN THE BASE UNIT - mm, degrees, MeV, which are Geant4's own - and a
/// unit here is a factor applied on the way in and out of a field. That matters because the
/// numbers a person has are not all in one unit: a CT resolution is quoted in mm, a phantom
/// in cm, a room in m, and a diagnostic beam in keV where a therapy beam is in MeV.
/// Converting by hand is where factor-of-ten mistakes come from.
struct UnitTable {
  const char* const* names = nullptr;
  const double* factors = nullptr;   ///< one of names[i] is factors[i] base units
  int n = 0;
  int base = 0;                      ///< index of the base unit, the default a field opens in
};

inline const UnitTable& LengthUnitTable() {
  static const char* const kNames[] = {"um", "mm", "cm", "m"};
  static const double kFactors[] = {0.001, 1.0, 10.0, 1000.0};
  static const UnitTable t{kNames, kFactors, 4, 1};
  return t;
}
inline const UnitTable& AngleUnitTable() {
  static const char* const kNames[] = {"deg", "rad", "mrad"};
  // 1 rad in degrees, written out rather than derived from a pi constant: it is exact to
  // more digits than a double carries, and this way there is one place to read it from.
  static const double kFactors[] = {1.0, 57.29577951308232, 0.05729577951308232};
  static const UnitTable t{kNames, kFactors, 3, 0};
  return t;
}
inline const UnitTable& EnergyUnitTable() {
  static const char* const kNames[] = {"eV", "keV", "MeV", "GeV", "TeV"};
  static const double kFactors[] = {1e-6, 1e-3, 1.0, 1e3, 1e6};
  static const UnitTable t{kNames, kFactors, 5, 2};
  return t;
}

/// The family a quantity quoted in @p base belongs to, or an empty table for a bare number.
///
/// Keyed off the unit string a parameter table already carries, so every shape's fields get
/// the right menu without a second table to keep in step with builder::ShapeParams - which
/// is the kind of hand-copied duplication docs/RISK.md has entries about. A unit this does
/// not know - "share", or the empty string on a count - gets no menu, which is right: there
/// is nothing to convert a fraction into.
inline const UnitTable& UnitsFor(const char* base) {
  static const UnitTable kNone{};
  if (base == nullptr) { return kNone; }
  if (std::strcmp(base, "mm") == 0) { return LengthUnitTable(); }
  if (std::strcmp(base, "deg") == 0) { return AngleUnitTable(); }
  if (std::strcmp(base, "MeV") == 0) { return EnergyUnitTable(); }
  return kNone;
}

inline double InUnit(double base_value, int unit_idx, const UnitTable& u) {
  if (u.n <= 0) { return base_value; }
  const int i = (unit_idx >= 0 && unit_idx < u.n) ? unit_idx : u.base;
  return base_value / u.factors[i];
}

inline bool EyeToggle(Context& ctx, int id, const Rect& r, bool on) {
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  const bool clicked = hover && ctx.in != nullptr && ctx.in->left_pressed;
  const int s = ctx.canvas.GlyphH() - 3;
  const int x0 = r.x + (r.w - s) / 2;
  const int y0 = r.y + (r.h - s) / 2;
  const int cx = x0 + s / 2;
  const int cy = y0 + s / 2;
  const Color line = on ? (hover ? theme::kText : theme::kAccent)
                        : (hover ? theme::kTextDim : theme::kBorder);
  // The lens: two triangles meeting at the corners, which is a rhombus and reads as an eye
  // once there is a pupil in it.
  if (on) {
    ctx.canvas.FillTriangle(x0, cy, cx, y0, x0 + s, cy, line);
    ctx.canvas.FillTriangle(x0, cy, cx, y0 + s, x0 + s, cy, line);
    // The pupil, in the panel's own colour, so it is a hole rather than a dot of paint.
    ctx.canvas.FillRect({cx - 1, cy - 1, 3, 3}, theme::kPanel);
  } else {
    // Outline only. Drawn as the same rhombus with a smaller one cut out of it, because
    // there is no triangle-stroke primitive and a two-pixel-thick outline at this size is
    // what a stroke would look like anyway.
    ctx.canvas.FillTriangle(x0, cy, cx, y0, x0 + s, cy, line);
    ctx.canvas.FillTriangle(x0, cy, cx, y0 + s, x0 + s, cy, line);
    ctx.canvas.FillTriangle(x0 + 2, cy, cx, y0 + 2, x0 + s - 2, cy, theme::kPanel);
    ctx.canvas.FillTriangle(x0 + 2, cy, cx, y0 + s - 2, x0 + s - 2, cy, theme::kPanel);
    ctx.canvas.HLine(x0, x0 + s, cy, line);
  }
  return clicked;
}

inline Rect ListRowSwatchRect(const Context& ctx, const Rect& r) {
  const int s = ctx.canvas.GlyphH() - 2;
  return Rect{r.x + 6, r.y + (r.h - s) / 2, s, s};
}

inline bool ListRow(Context& ctx, int id, const Rect& r, const std::string& label,
                    bool selected, Color swatch = 0) {
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  const bool clicked = hover && ctx.in->left_pressed;
  if (selected) {
    ctx.canvas.FillRect(r, rgba(86, 156, 214, 60));
  } else if (hover) {
    ctx.canvas.FillRect(r, rgba(255, 255, 255, 14));
  }
  int x = r.x + 6;
  if (swatch != 0) {
    const Rect sw = ListRowSwatchRect(ctx, r);
    ctx.canvas.FillRect(sw, swatch);
    ctx.canvas.StrokeRect(sw, theme::kBorder);
    x += sw.w + 8;
  }
  ctx.canvas.Text(x, r.y + (r.h - ctx.canvas.font->glyph_h) / 2, label,
                  selected ? theme::kText : theme::kTextDim);
  return clicked;
}

/// A list row whose label becomes an editable field when it is already selected and clicked
/// again - the rename gesture a file manager uses.
///
/// Returns 1 when the row was clicked, 2 when the name was committed, 0 otherwise. The caller
/// owns the buffer, so the text survives across frames without this widget keeping state.
inline int ListRowRenamable(Context& ctx, int id, const Rect& r, const std::string& label,
                            bool selected, bool editing, std::string& buffer,
                            Color swatch = 0) {
  if (editing) {
    // The field takes the whole row, swatch included: a rename is a text box, and a colour
    // chip beside a text box is one more thing to click by accident while typing.
    if (TextField(ctx, id, r, buffer, label, 40)) { return 2; }
    return 0;
  }
  return ListRow(ctx, id, r, label, selected, swatch) ? 1 : 0;
}

/// A scrolling text log. `scroll` is the first visible line; a negative value pins to the end,
/// which is what a console wants by default.
inline void TextLog(Context& ctx, const Rect& r, const std::vector<std::string>& lines,
                    int& scroll) {
  Panel(ctx, r, theme::kField);
  const int lh = ctx.canvas.font->glyph_h + 2;
  const int visible = (r.h - 8) / lh;
  const int total = static_cast<int>(lines.size());
  if (scroll < 0 || scroll > total - visible) { scroll = total - visible; }
  if (scroll < 0) { scroll = 0; }

  if (ctx.Hovering(r) && ctx.in->wheel != 0) {
    scroll -= ctx.in->wheel * 3;
    if (scroll < 0) { scroll = 0; }
    if (scroll > total - visible) { scroll = (total > visible) ? total - visible : 0; }
  }

  const Rect saved = ctx.canvas.clip;
  ctx.canvas.clip = r.Inset(2);
  for (int i = 0; i < visible; ++i) {
    const int li = scroll + i;
    if (li < 0 || li >= total) { continue; }
    const std::string& s = lines[li];
    // Color by severity, so a FATAL in a long log is findable.
    Color c = theme::kTextDim;
    if (s.find("FATAL") != std::string::npos || s.find("error") != std::string::npos) {
      c = theme::kError;
    } else if (s.find("WARNING") != std::string::npos) {
      c = theme::kWarn;
    } else if (s.find("sigma") != std::string::npos || s.find("Dose") != std::string::npos
               || s.find("dose") != std::string::npos) {
      c = theme::kOk;
    } else if (!s.empty() && s[0] == '/') {
      c = theme::kAccent;  // an echoed macro command
    }
    ctx.canvas.Text(r.x + 5, r.y + 4 + i * lh, s, c);
  }
  ctx.canvas.clip = saved;

  // Scrollbar, only when it would do something.
  if (total > visible && visible > 0) {
    const int track_h = r.h - 4;
    const int thumb_h = (track_h * visible) / total;
    const int thumb_y = r.y + 2 + (track_h - thumb_h) * scroll
                        / ((total - visible) > 0 ? (total - visible) : 1);
    ctx.canvas.FillRect({r.x + r.w - 6, thumb_y, 4, thumb_h > 8 ? thumb_h : 8},
                        theme::kBorder);
  }
}

/// A progress bar with a caption inside it.
inline void ProgressBar(Context& ctx, const Rect& r, double fraction,
                        const std::string& caption) {
  const double f = fraction < 0 ? 0 : (fraction > 1 ? 1 : fraction);
  ctx.canvas.FillRect(r, theme::kField);
  ctx.canvas.FillRect({r.x, r.y, static_cast<int>(f * r.w), r.h}, theme::kAccent);
  ctx.canvas.StrokeRect(r, theme::kBorder);
  ctx.canvas.TextCentred(r, caption, theme::kText);
}

// ---------------------------------------------------------------- menus

/// A menu bar item. Returns true when its dropdown should be shown.
struct MenuBar {
  int open = -1;        ///< index of the open menu, -1 for none
  int hovered_item = -1;
  int next_index = 0;
  Rect bar{};
  int pen_x = 0;
  Rect dropdown{};      ///< where the open menu is drawing
  int dropdown_pen = 0;
  int dropdown_index = 0;

  void Begin(Context& ctx, const Rect& r) {
    bar = r;
    pen_x = r.x + 4;
    next_index = 0;
    hovered_item = -1;
    Panel(ctx, r, theme::kPanelAlt);
  }

  /// A top-level menu title. Returns true when its dropdown is open.
  bool Menu(Context& ctx, const std::string& title) {
    const int w = ctx.canvas.font->TextWidth(title) + 20;
    const Rect r{pen_x, bar.y, w, bar.h};
    const int idx = next_index++;
    const bool hover = ctx.Hovering(r);
    if (hover && ctx.in->left_pressed) { open = (open == idx) ? -1 : idx; }
    if (hover && open >= 0 && open != idx) { open = idx; }  // slide across an open bar
    if (open == idx) {
      ctx.canvas.FillRect(r, theme::kAccent);
    } else if (hover) {
      ctx.canvas.FillRect(r, rgba(255, 255, 255, 20));
    }
    ctx.canvas.TextCentred(r, title, theme::kText);
    pen_x += w;
    if (open == idx) {
      // Wide enough for the widest label this menu had *last* frame, and never off the right
      // edge of the window. Measuring this frame is not possible: the width has to be fixed
      // before the items declare their hit rects, and the items are what set the width. One
      // frame of lag on a menu that has to be opened to be seen is invisible - the same trade
      // ScrollArea makes for its content height.
      int dw = (menu_w_[idx & 7] > 0) ? menu_w_[idx & 7] : 220;
      if (r.x + dw > ctx.canvas.width) { dw = ctx.canvas.width - r.x; }
      if (dw < 80) { dw = 80; }
      dropdown = {r.x, r.y + r.h, dw, 0};
      dropdown_pen = dropdown.y + 4;
      dropdown_index = 0;
      return true;
    }
    return false;
  }

  /// One dropdown entry. Call between Menu() returning true and EndMenu().
  bool Item(Context& ctx, const std::string& label, bool enabled = true) {
    const int h = ctx.canvas.font->glyph_h + 8;
    const Rect r{dropdown.x, dropdown_pen, dropdown.w, h};
    dropdown_pen += h;
    ++dropdown_index;
    const bool hover = enabled && ctx.Hovering(r);
    // The dropdown is drawn after the rest of the frame, so record and paint in EndMenu.
    items_.push_back({label, r, enabled, hover});
    if (hover && ctx.in->left_pressed) {
      open = -1;
      return true;
    }
    return false;
  }

  /// A submenu that opens laterally. Returns true while it is open, so the caller can emit
  /// its items with SubItem().
  ///
  /// Lateral rather than a separate window: a dialog for "which primitive?" is a modal
  /// interruption for a choice that is one click deep, and every other menu on the machine
  /// does this. The submenu opens on hover, as menus do, and stays open while the cursor is
  /// over either the parent row or the submenu itself - which is the part that has to be got
  /// right, or the panel closes as the cursor travels between them.
  bool Submenu(Context& ctx, const std::string& label) {
    const int h = ctx.canvas.font->glyph_h + 8;
    const Rect r{dropdown.x, dropdown_pen, dropdown.w, h};
    dropdown_pen += h;
    const int idx = ++dropdown_index;
    const bool over_row = ctx.Hovering(r);
    if (over_row) { sub_open = idx; }
    const bool open_here = (sub_open == idx);
    items_.push_back({label + "    >", r, true, over_row});
    if (open_here) {
      int sw = (sub_w_[idx & 15] > 0) ? sub_w_[idx & 15] : 200;
      int sx = r.x + r.w - 4;
      // Flip to the left of the parent when there is no room on the right, which is where a
      // submenu of a right-hand menu always is.
      if (sx + sw > ctx.canvas.width) { sx = r.x - sw + 4; }
      if (sx < 0) { sx = 0; }
      sub_rect = {sx, r.y, sw, 0};
      sub_pen = sub_rect.y + 4;
      sub_parent = r;
    }
    return open_here;
  }

  /// One entry of the open submenu.
  bool SubItem(Context& ctx, const std::string& label, bool enabled = true) {
    const int h = ctx.canvas.font->glyph_h + 8;
    const Rect r{sub_rect.x, sub_pen, sub_rect.w, h};
    sub_pen += h;
    const bool hover = enabled && ctx.Hovering(r);
    if (hover) { sub_hovered = true; }
    sub_items_.push_back({label, r, enabled, hover});
    if (hover && ctx.in->left_pressed) {
      open = -1;
      sub_open = -1;
      return true;
    }
    return false;
  }

  /// Closes the submenu unless the cursor is on it or on its parent row.
  void EndSubmenu(Context& ctx) {
    if (sub_open >= 0 && !sub_hovered
        && !sub_parent.Contains(ctx.in->mouse_x, ctx.in->mouse_y)) {
      const Rect box{sub_rect.x, sub_rect.y, sub_rect.w, sub_pen - sub_rect.y + 4};
      if (!box.Contains(ctx.in->mouse_x, ctx.in->mouse_y)) { sub_open = -1; }
    }
    sub_hovered = false;
  }

  void Separator(Context& ctx) {
    (void)ctx;
    items_.push_back({"", {dropdown.x, dropdown_pen, dropdown.w, 5}, false, false});
    dropdown_pen += 5;
  }

  /// Paints the recorded dropdown, then the submenu on top of it. Must be called after the
  /// items.
  void EndMenu(Context& ctx) {
    if (items_.empty()) {
      sub_items_.clear();
      return;
    }
    const Rect box{dropdown.x, dropdown.y, dropdown.w, dropdown_pen - dropdown.y + 4};
    ctx.canvas.FillRect(box, theme::kPanel);
    ctx.canvas.StrokeRect(box, theme::kBorder);
    int want = 0;
    for (const Entry& e : items_) {
      const int need = ctx.canvas.TextWidth(e.label) + 12 + 14;
      if (need > want) { want = need; }
      if (e.label.empty()) {
        ctx.canvas.HLine(e.r.x + 6, e.r.x + e.r.w - 6, e.r.y + 2, theme::kBorder);
        continue;
      }
      if (e.hover) { ctx.canvas.FillRect(e.r, theme::kAccent); }
      // Ellipsized as well as measured: the width is last frame's, so the very first frame a
      // menu is opened - or the frame after its labels grow - would otherwise draw a label
      // past the panel edge.
      ctx.canvas.TextFit(e.r.x + 12, e.r.y + 4, e.label,
                         e.enabled ? theme::kText : theme::kTextDim, e.r.w - 12 - 6);
    }
    if (open >= 0) { menu_w_[open & 7] = want; }
    items_.clear();
    // The submenu after the parent panel, so it draws over it rather than under.
    if (!sub_items_.empty()) {
      const Rect sbox{sub_rect.x, sub_rect.y, sub_rect.w, sub_pen - sub_rect.y + 4};
      ctx.canvas.FillRect(sbox, theme::kPanel);
      ctx.canvas.StrokeRect(sbox, theme::kAccent);
      int swant = 0;
      for (const Entry& e : sub_items_) {
        const int need = ctx.canvas.TextWidth(e.label) + 12 + 14;
        if (need > swant) { swant = need; }
        if (e.hover) { ctx.canvas.FillRect(e.r, theme::kAccent); }
        ctx.canvas.TextFit(e.r.x + 12, e.r.y + 4, e.label,
                           e.enabled ? theme::kText : theme::kTextDim, e.r.w - 12 - 6);
      }
      if (sub_open >= 0) { sub_w_[sub_open & 15] = swant; }
      sub_items_.clear();
    }
  }

  /// Closes any open menu when the user clicks elsewhere.
  void EndBar(Context& ctx) {
    if (open >= 0 && ctx.in->left_pressed && ctx.hot == 0
        && !bar.Contains(ctx.in->mouse_x, ctx.in->mouse_y)) {
      open = -1;
      sub_open = -1;
    }
    if (open < 0) { sub_open = -1; }
  }

 private:
  struct Entry {
    std::string label;
    Rect r;
    bool enabled;
    bool hover;
  };
  std::vector<Entry> items_;
  std::vector<Entry> sub_items_;
  int sub_open = -1;      ///< dropdown_index of the open submenu, -1 for none
  Rect sub_rect{};
  Rect sub_parent{};
  int sub_pen = 0;
  bool sub_hovered = false;
  // The width each dropdown wanted last frame, per menu / submenu index. See Menu().
  int menu_w_[8] = {};
  int sub_w_[16] = {};
};

// ---------------------------------------------------------------- dropdown

/// A dropdown: a button showing the current choice, with a list that opens below it.
///
/// The list cannot be drawn where the widget is declared, because everything drawn after it
/// this frame - the rest of the panel, the next section - would paint over it. So the widget
/// records that it is open and DrawOpenSelect() paints and hit-tests the list at the end of
/// the frame, after the panels and before the pop-ups.
///
/// That defers the click by one frame: the row is clicked while the list is being drawn late
/// in frame N, and the Select that owns it is not reached again until frame N+1. Hence
/// `select_picked`, which carries the row across. A frame at 60 Hz is 16 ms and the change
/// lands with the list closing, so there is nothing to see.
///
/// @return true when the value changed
/// @param enabled false draws the current value greyed and ignores clicks. For a menu whose
///        choice has stopped mattering - one overridden by something else the dialog was
///        given - which has to look inert rather than merely be inert.
inline bool Select(Context& ctx, int id, const Rect& r, int& value, const char* const* opts,
                   int count, bool enabled = true) {
  if (!enabled) {
    ctx.canvas.FillRect(r, theme::kField);
    ctx.canvas.StrokeRect(r, theme::kBorder);
    if (value >= 0 && value < count) {
      ctx.canvas.TextFit(r.x + 6, r.y + (r.h - ctx.canvas.GlyphH()) / 2, opts[value],
                         theme::kBorder, r.w - 12);
    }
    return false;
  }
  const bool hover = ctx.Hovering(r);
  if (hover) { ctx.hot = id; }
  const bool open = (ctx.open_select == id);

  // While open, the geometry is refreshed EVERY frame rather than kept from the frame the
  // list was opened on. A panel that scrolls, a splitter that moves or a window that resizes
  // moves the button, and a list pinned to where the button used to be is worse than one that
  // never opened. It also means a caller can open a list by id alone - the selftest does, to
  // photograph one inside a pop-up - without having to know its rectangle.
  if (open) {
    ctx.select_layer = ctx.layer;
    ctx.select_rect = r;
    ctx.select_opts = opts;
    ctx.select_count = count;
  }

  if (open && ctx.select_picked >= 0) {
    const int picked = ctx.select_picked;
    ctx.select_picked = -1;
    ctx.open_select = 0;
    if (picked < count && picked != value) {
      value = picked;
      return true;
    }
    return false;
  }
  if (hover && ctx.in != nullptr && ctx.in->left_pressed) {
    if (open) {
      ctx.open_select = 0;
    } else {
      ctx.open_select = id;
      ctx.select_layer = ctx.layer;
      ctx.select_rect = r;
      ctx.select_opts = opts;
      ctx.select_count = count;
      ctx.select_picked = -1;
    }
  }

  ctx.canvas.FillRect(r, open ? theme::kPanelAlt : theme::kField);
  ctx.canvas.StrokeRect(r, open ? theme::kAccent : (hover ? theme::kAccentHot
                                                          : theme::kBorder));
  const char* label = (value >= 0 && value < count && opts[value] != nullptr) ? opts[value]
                                                                             : "";
  const int ty = r.y + (r.h - ctx.canvas.GlyphH()) / 2;
  // The arrow gets its own reserved column, so a long option ellipsizes instead of running
  // into it.
  ctx.canvas.TextFit(r.x + 6, ty, label, theme::kText, r.w - 6 - 16);
  ctx.canvas.Text(r.x + r.w - 13, ty, "v", theme::kTextDim);
  return false;
}

/// Sets Context::layer for a scope and puts it back, so a return or an early exit inside a
/// pop-up cannot leave the layer raised for whatever draws next.
struct LayerScope {
  Context& ctx;
  int saved;
  LayerScope(Context& c, int layer) : ctx(c), saved(c.layer) { ctx.layer = layer; }
  ~LayerScope() { ctx.layer = saved; }
  LayerScope(const LayerScope&) = delete;
  LayerScope& operator=(const LayerScope&) = delete;
};

/// Paints and hit-tests the open dropdown's list, if it belongs to `layer`.
///
/// Call once per layer, immediately after everything at that layer is drawn: after the panels
/// with kLayerPanel, after the pop-ups with kLayerPopup. The layer test is what keeps a
/// dropdown declared inside a pop-up from being painted at the panel layer and then covered by
/// the pop-up that owns it - which is what happened for the whole life of the voxel import
/// dialog - and equally keeps a panel's dropdown from floating over a pop-up in front of it.
/// A number field with a unit menu beside it. Returns true when the VALUE changed.
///
/// @p base is the quantity in the base unit, which is what the model stores; the field shows
/// it divided by the chosen unit's factor.
///
/// CHANGING THE UNIT RE-DISPLAYS THE SAME QUANTITY rather than reinterpreting the number, so
/// mm to cm turns 50 into 5 and back into 50. The other reading - keep the number, change
/// what it means - moves the geometry every time the menu is touched, and does it silently,
/// which is the worst way for a control to be destructive.
///
/// @p unit_idx is held by the caller and outlives the selection, so someone working in cm
/// goes on working in cm as they click from one solid to the next.
inline bool UnitField(Context& ctx, int id, const Rect& fr, const Rect& ur, NumberField& f,
                      double& base, int& unit_idx, const UnitTable& u) {
  if (u.n <= 0) {
    if (NumberInput(ctx, id, fr, f)) {
      base = f.value;
      return true;
    }
    return false;
  }
  if (unit_idx < 0 || unit_idx >= u.n) { unit_idx = u.base; }
  if (Select(ctx, id + 40000, ur, unit_idx, u.names, u.n)) {
    f.Init(base / u.factors[unit_idx]);   // the same quantity, said differently
  }
  if (NumberInput(ctx, id, fr, f)) {
    base = f.value * u.factors[unit_idx];
    return true;
  }
  return false;
}

/// What @p base_value shows as in the unit at @p unit_idx. For seeding a field.
inline void DrawOpenSelect(Context& ctx, int layer = Context::kLayerPanel) {
  if (ctx.open_select == 0 || ctx.select_opts == nullptr || ctx.select_count <= 0) { return; }
  if (ctx.select_layer != layer) { return; }
  const Rect& b = ctx.select_rect;
  const int h = ctx.canvas.GlyphH() + 6;
  int w = b.w;
  for (int i = 0; i < ctx.select_count; ++i) {
    if (ctx.select_opts[i] == nullptr) { continue; }
    const int need = ctx.canvas.TextWidth(ctx.select_opts[i]) + 20;
    if (need > w) { w = need; }
  }
  if (b.x + w > ctx.canvas.width) { w = ctx.canvas.width - b.x; }

  // Below by default; above when there is no room, so a dropdown at the foot of a panel is
  // still usable rather than opening off the bottom of the window.
  int y = b.y + b.h;
  const int list_h = ctx.select_count * h + 4;
  if (y + list_h > ctx.canvas.height) {
    const int above = b.y - list_h;
    y = (above >= 0) ? above : ctx.canvas.height - list_h;
  }

  // The list is drawn over whatever is there, so its clip has to be the whole canvas: the
  // panel clip that was in force where the Select was declared has long since been restored,
  // but a caller may have left one set.
  const Rect saved = ctx.canvas.clip;
  ctx.canvas.clip = {0, 0, ctx.canvas.width, ctx.canvas.height};
  const Rect box{b.x, y, w, list_h};
  ctx.canvas.FillRect(box, theme::kPanel);
  ctx.canvas.StrokeRect(box, theme::kAccent);
  bool over_list = false;
  for (int i = 0; i < ctx.select_count; ++i) {
    const Rect row{box.x + 2, y + 2 + i * h, box.w - 4, h};
    const bool hover = ctx.Hovering(row);
    if (hover) {
      over_list = true;
      ctx.canvas.FillRect(row, theme::kAccent);
    }
    const char* s = (ctx.select_opts[i] != nullptr) ? ctx.select_opts[i] : "";
    ctx.canvas.TextFit(row.x + 8, row.y + 3, s, theme::kText, row.w - 12);
    if (hover && ctx.in != nullptr && ctx.in->left_pressed) { ctx.select_picked = i; }
  }
  ctx.canvas.clip = saved;

  // A click anywhere else closes it without choosing. Not `ctx.hot == 0`, as the menu bar
  // does: the list overlaps other widgets, and one of them will have claimed hot on the way
  // past - the list is drawn last, so it never gets to claim it itself.
  if (!over_list && ctx.in != nullptr && ctx.in->left_pressed
      && !b.Contains(ctx.in->mouse_x, ctx.in->mouse_y)) {
    ctx.open_select = 0;
  }
}

}  // namespace g4gpu::ui
