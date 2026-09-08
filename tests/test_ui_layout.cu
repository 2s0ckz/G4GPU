// How a panel's height is divided, and what a scroll area's bar geometry is.
//
// WHY THIS EXISTS
//
// Each panel section scrolls in its own box, because one scroll area per panel put every
// section on one scrollbar and a segmented phantom's few hundred voxel classes pushed SOURCES
// out of reach. That made "how tall is each box" a policy.
//
// The policy used to be ui::DivideSections, which shared the height out in proportion to what
// each section asked for, and this file used to test that. It is gone: a panel whose sections
// move as the model grows is one where nothing is ever twice in the same place, so the
// sidebars are fixed halves and thirds now. The arithmetic that remains is the split between
// a LIST and the FORM below it - ui::SplitListForm - which is what this covers, along with
// the scrollbar geometry that ui::ScrollArea::BarRects hands to both the drag and the paint.
//
// Both have invariants worth pinning rather than eyeballing:
//
//   * the two rects must exactly fill the area they were given, with no overlap - a pixel of
//     slack between two scroll regions is a gap or a double-clipped row;
//   * the list must never be padded past its own content - a band of nothing between a
//     short list and its form is what a fixed split produced - and the form must always be
//     owed what it asked for, up to half the section;
//   * the thumb must stay inside its track at every scroll offset, and must reach both ends -
//     a thumb that stops short cannot scroll to the bottom by dragging;
//   * and a widget clipped out of its section must not be clickable, which is a property of
//     Context::Hovering rather than of any widget - see section 8. The same goes for a widget
//     an open dropdown is painted over, which is section 11.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "render/ui.h"

using namespace g4gpu;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

/// The invariants of a split, for any input.
void CheckSplit(const ui::Rect& area, int want_list, int want_form, const char* what) {
  ui::Rect list, form;
  ui::SplitListForm(area, want_list, want_form, list, form);
  char buf[180];
  std::snprintf(buf, sizeof buf, "%s: the two rects fill the area exactly", what);
  Check(list.h + form.h == area.h, buf);
  std::snprintf(buf, sizeof buf, "%s: the form starts where the list ends", what);
  Check(form.y == list.y + list.h, buf);
  std::snprintf(buf, sizeof buf, "%s: neither is negative", what);
  Check(list.h >= 0 && form.h >= 0, buf);
  // The form is owed what it asked for, up to half the section. This is the invariant that
  // was wrong first time round, and it showed up as a three-row solid list.
  const int share = area.h / 2;
  const int owed = (want_form < share) ? want_form : share;
  std::snprintf(buf, sizeof buf, "%s: the form gets what it needs or half, whichever is less",
                what);
  Check(form.h >= owed, buf);
  // And the list is never given more than it asked for, which is what closes the band of
  // nothing between a short list and the form under it.
  std::snprintf(buf, sizeof buf, "%s: the list is not padded past its content", what);
  Check(list.h <= (want_list > 0 ? want_list : 0), buf);
}

}  // namespace

int main() {
  std::printf("== how a panel's height is divided ==\n\n");

  const int row = 19;   // a typical glyph height plus padding

  // ---- 1. A short list and a form: THE FORM BEGINS UNDER THE LAST ROW.
  //
  // Which is the defect this signature exists to fix. A fixed split - the form anchored to
  // the bottom of its section - left a band of nothing between a two-source list and the form
  // below it, and the band grew as the panel did.
  {
    const ui::Rect area{0, 100, 300, 600};
    ui::Rect list, form;
    ui::SplitListForm(area, 2 * row, 220, list, form);
    Check(list.h == 2 * row, "a two-row list is two rows tall and no taller");
    Check(form.y == 100 + 2 * row, "and the form begins immediately under it");
    Check(form.h == 600 - 2 * row, "taking everything below");
    CheckSplit(area, 2 * row, 220, "a short list");
  }

  // ---- 2. No selection, so no form: the list may have the whole section.
  {
    const ui::Rect area{0, 0, 300, 600};
    ui::Rect list, form;
    ui::SplitListForm(area, 900, 0, list, form);
    Check(list.h == 600 && form.h == 0, "with nothing selected the list gets the section");
  }

  // ---- 3. The case the first version got wrong: a long list AND a long form.
  //
  // Eleven solids and a voxel volume's form - a dozen fields and six buttons - is this. A
  // floor of three rows for the list gave it three rows; half the section gives it half.
  {
    const ui::Rect area{0, 0, 300, 400};
    ui::Rect list, form;
    ui::SplitListForm(area, 900, 900, list, form);
    Check(list.h == 200, "a long list against a long form gets half the section");
    Check(form.h == 200, "and so does the form");
    CheckSplit(area, 900, 900, "both over-long");
  }

  // ---- 4. A long list against a SHORT form: the form keeps what it needs and the list takes
  //         the remainder rather than half. Half each would waste the space in a section
  //         whose form is four rows.
  {
    const ui::Rect area{0, 0, 300, 400};
    ui::Rect list, form;
    ui::SplitListForm(area, 900, 60, list, form);
    Check(form.h == 60, "a short form gets exactly what it needs");
    Check(list.h == 340, "and the list gets the whole remainder");
    CheckSplit(area, 900, 60, "a short form");
  }

  // ---- 5. A section dragged down to nothing. The invariant that survives is that the two
  //         rects still tile it.
  {
    for (int h = 0; h <= 4 * row; ++h) {
      const ui::Rect area{0, 0, 300, h};
      char what[64];
      std::snprintf(what, sizeof what, "a %d-pixel section", h);
      CheckSplit(area, 500, 500, what);
    }
  }

  // ---- 6. A sweep, because the invariants have to hold for every shape of input and not
  //         only the five cases someone thought of.
  {
    bool ok = true;
    for (int h = 1; h <= 1200; h += 7) {
      for (int wl = 0; wl <= 2000; wl += 91) {
        for (int wf = 0; wf <= 2000; wf += 91) {
          ui::Rect list, form;
          const ui::Rect area{0, 40, 300, h};
          ui::SplitListForm(area, wl, wf, list, form);
          if (list.h + form.h != h) { ok = false; }
          if (form.y != list.y + list.h) { ok = false; }
          if (list.h < 0 || form.h < 0) { ok = false; }
          if (list.h > wl) { ok = false; }
          const int share = h / 2;
          const int owed = (wf < share) ? wf : share;
          if (form.h < owed) { ok = false; }
        }
      }
    }
    Check(ok, "the split tiles the section, never pads the list, and always owes the form its"
              " share - for every height and both demands");
  }

  // ---- 7. The scrollbar. Its geometry is shared by the drag (in Begin) and the paint (in
  //         End), and a thumb drawn anywhere other than where the drag picks it up is a
  //         control that jumps out from under the cursor.
  {
    ui::ScrollArea sa;
    const ui::Rect r{0, 50, 300, 200};
    ui::Rect track, thumb;

    sa.content_h = 150;   // fits
    sa.BarRects(r, track, thumb);
    Check(track.h == 0 && thumb.h == 0, "content that fits has no bar at all");

    sa.content_h = 800;
    sa.offset = 0;
    sa.BarRects(r, track, thumb);
    Check(track.h == r.h, "the track spans the view");
    Check(track.x + track.w == r.x + r.w,
          "and sits at the right edge, in the panel's own margin");
    Check(thumb.y == r.y, "at offset zero the thumb is at the top");
    Check(thumb.h > 0 && thumb.h < r.h, "and is shorter than the track it moves in");

    // The far end. max_off is content minus view; the thumb has to reach the bottom exactly
    // there, because anything less means the last rows cannot be reached by dragging.
    sa.offset = 800 - 200;
    sa.BarRects(r, track, thumb);
    Check(thumb.y + thumb.h == r.y + r.h, "at full scroll the thumb reaches the bottom");

    bool inside = true;
    for (int off = 0; off <= 600; ++off) {
      sa.offset = off;
      sa.BarRects(r, track, thumb);
      if (thumb.y < r.y || thumb.y + thumb.h > r.y + r.h) { inside = false; }
    }
    Check(inside, "and stays inside the track at every offset between");

    // A very long list. The thumb has a floor, because a strictly proportional thumb over a
    // few hundred thousand pixels of rows is sub-pixel and cannot be grabbed at all.
    sa.content_h = 400000;
    sa.offset = 0;
    sa.BarRects(r, track, thumb);
    Check(thumb.h >= 16, "a huge list still has a grabbable thumb");
    sa.offset = 400000 - 200;
    sa.BarRects(r, track, thumb);
    Check(thumb.y + thumb.h == r.y + r.h,
          "and the floored thumb still reaches the bottom rather than stopping short");
  }

  // ---- 8. A WIDGET YOU CANNOT SEE MUST NOT BE CLICKABLE.
  //
  // Every widget in this UI asks Context::Hovering whether the cursor is on it. Hovering used
  // to answer from the widget's rectangle alone, and the canvas clip - which is what stops a
  // scrolled-away row from being painted - had no say in it. So a button scrolled out of the
  // bottom of SOLIDS was invisible and still live, floating over the SOURCES section below,
  // and clicking a source opened the material picker for a solid that was not on screen.
  //
  // The fix is one line in Hovering, and it is one line rather than a change in every widget
  // because there is no widget for which "clickable but not visible" is right.
  {
    ui::Input in;
    ui::Context ctx;
    ctx.in = &in;

    // A section 200 tall, and a button 40 pixels below its bottom edge - what the last button
    // of a scrolled list looks like once the list is long enough to push it out.
    const ui::Rect section{1200, 100, 300, 200};
    const ui::Rect button{1210, 340, 200, 24};

    ctx.canvas.clip = ui::Rect{0, 0, 1600, 900};   // no clipping: the widget is live
    in.mouse_x = 1250;
    in.mouse_y = 350;
    Check(ctx.Hovering(button), "with no clip the cursor is on the button");

    ctx.canvas.clip = section;                     // the section's box, as ScrollArea sets it
    Check(!ctx.Hovering(button),
          "clipped out of its section, the same button is not hovered");

    // And the row that IS visible in that section still is, so the fix has not made the
    // clipping over-eager.
    const ui::Rect visible_row{1210, 150, 200, 24};
    in.mouse_x = 1250;
    in.mouse_y = 160;
    Check(ctx.Hovering(visible_row), "a row inside the section is still hovered");

    // The boundary itself: the last pixel row of the section is inside, the first below is
    // not. Off by one here is a one-pixel strip of dead or live widget along every section
    // edge in the program.
    in.mouse_y = 299;
    Check(ctx.Hovering(ui::Rect{1210, 290, 200, 24}), "the last pixel of the section counts");
    in.mouse_y = 300;
    Check(!ctx.Hovering(ui::Rect{1210, 290, 200, 24}),
          "and the first pixel past it does not");

    // A null input is not hovering anything, which is what a headless frame looks like.
    ctx.in = nullptr;
    Check(!ctx.Hovering(visible_row), "with no input nothing is hovered");
  }

  // ---- 9. The same thing through a real ScrollArea, since that is what sets the clip.
  //
  // Begin narrows the clip to the section and returns the y to draw at; End puts the clip
  // back. A widget drawn between them at a y past the section's bottom is exactly the case
  // above, and this checks the two agree rather than assuming they do.
  {
    ui::Input in;
    ui::Context ctx;
    ctx.in = &in;
    ctx.canvas.clip = ui::Rect{0, 0, 1600, 900};

    ui::ScrollArea sa;
    const ui::Rect section{1200, 100, 300, 200};
    sa.content_h = 900;      // a long list, so there is something to scroll
    sa.offset = 0;

    const int y0 = sa.Begin(ctx, section, 30, 9001);
    // Two rows: one inside the section, one well past its bottom.
    const ui::Rect inside{1210, y0 + 10, 200, 24};
    const ui::Rect below{1210, y0 + 400, 200, 24};
    in.mouse_x = 1250;
    in.mouse_y = inside.y + 5;
    const bool hit_inside = ctx.Hovering(inside);
    in.mouse_y = below.y + 5;
    const bool hit_below = ctx.Hovering(below);
    sa.End(ctx, section, y0 + 900);

    Check(hit_inside, "inside the scroll area, a visible row is hovered");
    Check(!hit_below, "and a row past the bottom of it is not");

    // End restored the clip, so a widget drawn after the section - the next section, or a
    // pop-up - is live again at those same coordinates.
    in.mouse_y = below.y + 5;
    Check(ctx.Hovering(below), "once the scroll area ends the clip is back and it is live");
  }

  // ---- 10. NO TWO WIDGETS MAY SHARE AN ID.
  //
  // The builder identifies each widget by an integer, and the lists that grow with the model
  // are numbered `base + index`. That scheme fails silently once a list reaches the next
  // base, and it had failed in four places at once:
  //
  //   * the solid rows started at 400 and the "Assign material" button was 410, so an
  //     eleven-solid model gave row 10 the button's id;
  //   * the source rows started at 690 against kind buttons at 700, the same at eleven;
  //   * elements at 200 and materials at 300 collided with their own controls at ten;
  //   * and the voxel classes were `4000 + solid*100 + class` while the importer caps
  //     classes at 4096 - so a phantom past a hundred classes collided WITH ITSELF, and the
  //     261-class segmentations this program was changed to accept are all past it.
  //
  // A shared id does not draw wrong. It shares ctx.hot, ctx.active and ctx.focus, so a rename
  // types into the wrong row and a drag is picked up by a control the cursor is nowhere near.
  // Nothing about that points at numbering.
  //
  // The blocks are a million apart now. This checks the arithmetic rather than the layout: it
  // enumerates the ids a large model would produce and looks for any repeat.
  {
    // The bases, copied from the enum in g4builder.cu. Copied deliberately: a test that
    // included the header would be asserting that a value equals itself. If the enum moves,
    // this fails and someone reads both.
    const int kElementRow = 1000000;
    const int kMaterialRow = 2000000;
    const int kScorerRow = 3000000;
    const int kSolidRow = 4000000;
    const int kSolidEye = 5000000;
    const int kClassRow = 6000000;
    const int kClassEye = 7000000;
    const int kSourceRow = 8000000;
    const int kPickRow = 9000000;
    const int kClassStride = 4096;
    // The importer's cap, from builder/import.hh. The stride must be at least this or a
    // solid's classes run into the next solid's block.
    const int kMaxClasses = 4096;
    Check(kClassStride >= kMaxClasses,
          "the class id stride is at least the importer's class cap");

    // A model far larger than anything anyone builds: 200 solids, each with a full 4096
    // classes, plus long lists of everything else.
    const int kSolids = 200;
    const int kClasses = 4096;
    const int kOthers = 5000;

    std::vector<long long> ids;
    ids.reserve(static_cast<std::size_t>(kSolids) * (kClasses * 2 + 2) + 4 * kOthers);
    for (int i = 0; i < kOthers; ++i) {
      ids.push_back(kElementRow + i);
      ids.push_back(kMaterialRow + i);
      ids.push_back(kScorerRow + i);
      ids.push_back(kSourceRow + i);
      ids.push_back(kPickRow + i);
    }
    for (int i = 0; i < kSolids; ++i) {
      ids.push_back(kSolidRow + i);
      ids.push_back(kSolidEye + i);
      for (int k = 0; k < kClasses; ++k) {
        ids.push_back(kClassRow + static_cast<long long>(i) * kClassStride + k);
        ids.push_back(kClassEye + static_cast<long long>(i) * kClassStride + k);
      }
    }
    // Every fixed control in the builder is below 50000 - the highest literal is the import
    // dialog's 1280 - so the blocks must all start above that and nothing may reach down.
    long long lowest = ids[0];
    for (long long q : ids) {
      if (q < lowest) { lowest = q; }
    }
    Check(lowest >= 50000, "no growing list reaches down into the fixed control numbers");

    std::sort(ids.begin(), ids.end());
    bool dup = false;
    long long first_dup = 0;
    for (std::size_t i = 1; i < ids.size(); ++i) {
      if (ids[i] == ids[i - 1] && !dup) {
        dup = true;
        first_dup = ids[i];
      }
    }
    if (dup) {
      std::printf("  first repeated id: %lld\n", first_dup);
    }
    Check(!dup, "200 solids of 4096 classes and 5000 of everything else share no id");

    // And the case that actually broke: the old scheme, checked to be sure this test would
    // have caught it rather than passing either way.
    std::vector<long long> old;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 200; ++k) { old.push_back(4000 + i * 100 + k); }
    }
    std::sort(old.begin(), old.end());
    bool old_dup = false;
    for (std::size_t i = 1; i < old.size(); ++i) {
      if (old[i] == old[i - 1]) { old_dup = true; }
    }
    Check(old_dup, "the old stride of 100 does collide at 200 classes, as claimed");
  }

  // ---- 11. AN OVERLAY PAINTED LAST HAS TO BE HIT-TESTED FIRST.
  //
  // An open dropdown's list is drawn after the panel that declared it, because everything
  // drawn after a widget paints over it. That puts its rows LAST in the frame, so by the time
  // they are tested every widget they cover has already had its turn at the cursor - and one
  // of them has taken the click. Reported as: picking a row of an open menu opens the closed
  // menu that the row happens to sit on top of.
  //
  // Context::block is the region the list occupied, carried over from the frame it was painted
  // on, and Hovering refuses the cursor to anything at or behind its layer inside it.
  {
    // The pure property first, with no widgets involved: it is one condition in Hovering, the
    // same as the clip in section 8, and the layer is what stops it over-reaching.
    ui::Input in;
    ui::Context ctx;
    ctx.in = &in;
    ctx.canvas.clip = ui::Rect{0, 0, 1600, 900};
    const ui::Rect covered{100, 200, 120, 24};
    in.mouse_x = 150;
    in.mouse_y = 210;
    Check(ctx.Hovering(covered), "with no list open the widget is hovered");

    ctx.block = ui::Rect{80, 180, 200, 100};
    ctx.block_layer = ui::Context::kLayerPanel;
    ctx.layer = ui::Context::kLayerPanel;
    Check(!ctx.Hovering(covered), "under an open list at its own layer, it is not");
    ctx.layer = ui::Context::kLayerPopup;
    Check(ctx.Hovering(covered),
          "a pop-up in front of the list is not covered by it and stays live");
    ctx.layer = ui::Context::kLayerPanel;
    in.mouse_x = 400;   // outside the list
    Check(ctx.Hovering(ui::Rect{380, 200, 120, 24}),
          "a widget beside the list is unaffected");
  }

  // And the reported case end to end, through the real widgets: two dropdowns in a column,
  // the upper one open, and a click on the row of its list that lies over the lower one's
  // button. The lower one must not open, and the row must be the thing that was picked.
  {
    // A font filled in rather than built: Font::Build is GDI, and what this needs from a font
    // is its METRICS. Fixed numbers also make the row arithmetic below the same on every
    // machine, where a real face's tmHeight is whatever the system decided.
    ui::Font font;
    font.glyph_w = 8;
    font.glyph_h = 15;
    font.ascent = 12;
    font.coverage.assign(
        static_cast<std::size_t>(ui::Font::kCount) * font.glyph_w * font.glyph_h, 0);
    font.ready = true;
    {
      constexpr int kW = 640, kH = 600;
      std::vector<unsigned int> px(static_cast<std::size_t>(kW) * kH, 0u);
      ui::Input in;
      ui::Context ctx;

      static const char* const kOpts[4] = {"mm", "cm", "m", "um"};
      const int rh = font.glyph_h + 6;          // DrawOpenSelect's row height
      const ui::Rect ra{100, 100, 120, 22};     // the upper dropdown
      // The middle of row 1 of the list the upper one opens: below its button, 2 px of
      // border, then one whole row.
      const int row1_mid = ra.y + ra.h + 2 + rh + rh / 2;
      const ui::Rect rb{100, row1_mid - 10, 120, 22};   // the lower dropdown, under that row
      const ui::Rect btn{100, row1_mid + rh - 10, 120, 22};  // and a plain button under row 2

      int va = 0, vb = 0;
      bool button_fired = false;
      // One frame of the panel: two dropdowns, a button, then the open list.
      auto frame = [&]() {
        ctx.Begin(px.data(), kW, kH, &font, &in);
        ctx.canvas.clip = ui::Rect{0, 0, kW, kH};
        ui::Select(ctx, 11, ra, va, kOpts, 4);
        ui::Select(ctx, 22, rb, vb, kOpts, 4);
        if (ui::Button(ctx, 33, btn, "Import")) { button_fired = true; }
        ui::DrawOpenSelect(ctx, ui::Context::kLayerPanel);
        ctx.End();
        in.EndFrame();
      };

      // Frame 1: click the upper dropdown's button. Its list opens and is painted.
      in.mouse_x = ra.x + 10;
      in.mouse_y = ra.y + 10;
      in.left_down = true;
      in.left_pressed = true;
      frame();
      Check(ctx.open_select == 11, "clicking the upper dropdown opens it");
      const ui::Rect list = ctx.block;
      Check(list.h > 0 && list.Contains(rb.x + 10, row1_mid),
            "its list covers the lower dropdown's button");

      // Frame 2: the mouse is on row 1, which is on top of the LOWER dropdown's button, and
      // clicks. This is the reported gesture.
      in.mouse_x = rb.x + 10;
      in.mouse_y = row1_mid;
      in.left_down = true;
      in.left_pressed = true;
      frame();
      Check(ctx.open_select == 11, "the click stays with the open dropdown");
      Check(ctx.select_picked == 1, "and it picked the row that was clicked");
      Check(vb == 0, "the covered dropdown's value is untouched");

      // Frame 3: the upper dropdown consumes the pick and closes.
      frame();
      Check(va == 1, "the value it was opened for changed");
      Check(ctx.open_select == 0, "and the list is closed");
      Check(ctx.block.w == 0 && ctx.block.h == 0, "with nothing left blocked");

      // A plain button under the list is no different: the same rule covers every widget,
      // not only the one whose id happened to be a dropdown.
      va = 0;
      in.mouse_x = ra.x + 10;
      in.mouse_y = ra.y + 10;
      in.left_down = true;
      in.left_pressed = true;
      frame();
      in.mouse_x = btn.x + 10;
      in.mouse_y = btn.y + 10;
      in.left_down = true;
      in.left_pressed = true;
      in.left_released = false;
      frame();
      in.left_down = false;
      in.left_released = true;
      frame();
      Check(!button_fired, "a button under the open list does not fire");

      // ---- 12. AND THE MENU BAR, which had the same fault and a worse symptom.
      //
      // MenuBar::Item records its rows and EndMenu paints them, and the bar draws after the
      // panels - so a panel control under an open menu had already claimed the click. Unlike
      // a dropdown, Item() returns true on hover-and-press whoever else took it, so a menu
      // entry over a panel button fired BOTH of them.
      //
      // The bar borrows the same one blocked region, so this also checks the save and restore
      // that keeps it from wiping an open dropdown's region on its way past.
      ui::MenuBar mb;
      bool menu_button_fired = false;
      bool item_fired = false;
      // The middle of the menu panel's third row. The panel opens at the foot of the bar with
      // 4 px of border, and each row is glyph_h + 8 tall - derived rather than guessed,
      // because the point of the test is that the button below is UNDER that row.
      const int item_y = 24 + 4 + 2 * (font.glyph_h + 8) + (font.glyph_h + 8) / 2;
      const ui::Rect under{20, item_y - 10, 160, 22};   // a panel control under that row
      // Button fires on RELEASE while it holds `active`, so a press-only frame proves nothing
      // about it - the check that the button underneath stays quiet has to run the release
      // too. Found by removing the fix and watching that assertion pass anyway.
      auto menu_frame = [&](bool press, bool release, int mx, int my) {
        ctx.Begin(px.data(), kW, kH, &font, &in);
        ctx.canvas.clip = ui::Rect{0, 0, kW, kH};
        in.mouse_x = mx;
        in.mouse_y = my;
        in.left_down = press;
        in.left_pressed = press;
        in.left_released = release;
        // The panel first, as the real frame does.
        if (ui::Button(ctx, 44, under, "Add")) { menu_button_fired = true; }
        mb.Begin(ctx, ui::Rect{0, 0, kW, 24});
        if (mb.Menu(ctx, "File")) {
          if (mb.Item(ctx, "New")) { item_fired = true; }
          if (mb.Item(ctx, "Open")) { item_fired = true; }
          if (mb.Item(ctx, "Save")) { item_fired = true; }
          mb.EndMenu(ctx);
        }
        mb.EndBar(ctx);
        ctx.End();
        in.EndFrame();
      };

      menu_frame(true, false, 20, 12);     // click the File title: the menu opens
      menu_frame(false, true, 20, 12);     // release it
      menu_frame(false, false, 20, 12);    // a frame with it open, so its panel is published
      Check(ctx.block.h > 0, "an open menu publishes the region its panel covers");
      Check(under.Contains(30, item_y) && ctx.block.Contains(30, item_y),
            "and that region reaches the button underneath");
      menu_frame(true, false, 30, item_y);
      menu_frame(false, true, 30, item_y);
      Check(item_fired, "the menu item fires");
      Check(!menu_button_fired, "and the button underneath it does not");

      // The bar must have put back the region it found, or an open dropdown loses its rows.
      in.left_down = false;
      ctx.open_select = 0;
      ctx.block = ui::Rect{};
      va = 0;
      in.mouse_x = ra.x + 10;
      in.mouse_y = ra.y + 10;
      in.left_down = true;
      in.left_pressed = true;
      ctx.Begin(px.data(), kW, kH, &font, &in);
      ctx.canvas.clip = ui::Rect{0, 0, kW, kH};
      ui::Select(ctx, 11, ra, va, kOpts, 4);
      ui::DrawOpenSelect(ctx, ui::Context::kLayerPanel);
      mb.Begin(ctx, ui::Rect{0, 0, kW, 24});   // no menu open: nothing to publish
      mb.EndBar(ctx);
      ctx.End();
      in.EndFrame();
      Check(ctx.block.h > 0,
            "a menu bar with nothing open leaves an open dropdown's region alone");
    }
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
