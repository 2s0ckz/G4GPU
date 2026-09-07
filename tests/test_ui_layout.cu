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
//     a thumb that stops short cannot scroll to the bottom by dragging.
#include <cstdio>

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

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
