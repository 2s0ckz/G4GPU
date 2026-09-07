// ui::DivideSections: how a sidebar's height is shared between its sections.
//
// WHY THIS EXISTS
//
// Each panel section scrolls in its own box now, because one scroll area per panel put every
// section on one scrollbar and a segmented phantom's few hundred voxel classes pushed SOURCES
// out of reach. That turned "how tall is each box" into a policy, and a policy with two
// invariants worth pinning: the boxes must exactly fill the panel - a pixel of slack is a gap
// or an overlap between two scroll regions - and a section squeezed by a greedy neighbour must
// still be big enough to use.
//
// The second one is the reason for the numbers below. Sharing the surplus in proportion to what
// each section asked for gives a 261-class list nearly everything, and the first version of
// this left SOURCES six rows to hold a header, a list, four buttons and eight fields.
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

}  // namespace

int main() {
  std::printf("== how a panel's height is shared between its sections ==\n\n");

  const int row = 19;   // a typical glyph height plus padding
  const int min_h = row * 5;

  // ---- 1. Everything fits: each section gets exactly what it asked for, and nothing is
  //         stretched to fill the panel. A section that wants four rows gets four rows.
  {
    const int want[2] = {200, 150};
    int h[2] = {0, 0};
    ui::DivideSections(900, min_h, want, h, 2);
    Check(h[0] == 200 && h[1] == 150, "two sections that fit keep their own heights");
  }

  // ---- 2. A tiny section still gets the floor, so a one-row list is clickable.
  {
    const int want[2] = {200, 4};
    int h[2] = {0, 0};
    ui::DivideSections(900, min_h, want, h, 2);
    Check(h[1] >= min_h, "a section asking for almost nothing still gets the floor");
  }

  // ---- 3. The case this was written for: 261 voxel classes against a short source list.
  {
    const int want[2] = {261 * row + 400, 300};   // ~5360 and 300
    int h[2] = {0, 0};
    const int avail = 900;
    ui::DivideSections(avail, min_h, want, h, 2);
    std::printf("  261 classes vs a short list, 900 px: %d / %d\n", h[0], h[1]);
    Check(h[0] + h[1] == avail, "the two boxes exactly fill the panel");
    Check(h[0] > h[1], "the long list gets the larger share");
    // Half a fair share is the floor, so the squeezed section keeps something workable. At
    // 900 pixels and two sections that is 225, about eleven rows.
    Check(h[1] >= avail / 4, "and the squeezed section keeps at least half a fair share");
  }

  // ---- 4. Three sections, which is the left sidebar.
  {
    const int want[3] = {4000, 120, 900};
    int h[3] = {0, 0, 0};
    const int avail = 870;
    ui::DivideSections(avail, min_h, want, h, 3);
    std::printf("  three sections, 870 px: %d / %d / %d\n", h[0], h[1], h[2]);
    Check(h[0] + h[1] + h[2] == avail, "three boxes exactly fill the panel");
    for (int i = 0; i < 3; ++i) {
      Check(h[i] >= avail / 6, "every one of three sections keeps half a fair share");
    }
  }

  // ---- 5. A panel dragged narrower than its own floors. Equal shares, exactly filling, and
  //         each section's own scrollbar does the rest - which is the honest answer rather
  //         than sections that overlap or run off the bottom.
  {
    const int want[3] = {900, 900, 900};
    int h[3] = {0, 0, 0};
    const int avail = 100;
    ui::DivideSections(avail, min_h, want, h, 3);
    Check(h[0] + h[1] + h[2] == avail, "boxes fill exactly even when the floors do not fit");
    for (int i = 0; i < 3; ++i) { Check(h[i] > 0, "and no section is given zero height"); }
  }

  // ---- 6. A sweep, because the two invariants have to hold for every shape of input and not
  //         only the ones I thought of. An overlap or a gap between scroll boxes is invisible
  //         in a screenshot, which is exactly why it is asserted here.
  {
    int bad_sum = 0, bad_zero = 0;
    for (int avail = 60; avail <= 1400; avail += 7) {
      for (int wa = 0; wa <= 6000; wa += 311) {
        for (int wb = 0; wb <= 6000; wb += 733) {
          const int want[2] = {wa, wb};
          int h[2] = {0, 0};
          ui::DivideSections(avail, min_h, want, h, 2);
          const int sum = h[0] + h[1];
          // Either everything fitted - in which case the boxes are their own size and may
          // leave the foot of the panel empty - or the panel is exactly filled.
          const bool fitted = (wa > min_h ? wa : min_h) + (wb > min_h ? wb : min_h) <= avail;
          if (!fitted && sum != avail) { ++bad_sum; }
          if (h[0] <= 0 || h[1] <= 0) { ++bad_zero; }
        }
      }
    }
    std::printf("  swept every shape: %d that did not fill the panel, %d with a zero box\n",
                bad_sum, bad_zero);
    Check(bad_sum == 0, "a divided panel is always filled exactly");
    Check(bad_zero == 0, "no input produces a zero-height section");
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
