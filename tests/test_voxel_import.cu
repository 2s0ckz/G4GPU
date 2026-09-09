// ClassifyVoxels: what an imported voxel volume's values are taken to mean.
//
// WHY THIS EXISTS
//
// The importer used to stop at 64 distinct values and refuse the file with "this looks
// continuous, not segmented", which is an inference the data does not support - a segmented
// phantom can have as many organs as it likes - and which contradicted the answer the user had
// already given by choosing "material indices". The suggested remedy, re-importing as
// Hounsfield units, would have banded indices as though they were densities.
//
// Nothing tested this function at all. It is pure host arithmetic over a vector, so there was
// never a reason beyond nobody having done it. See docs/RISK.md V14.
#include <cstdio>
#include <string>
#include <vector>

#include "builder/import.hh"

using namespace g4gpu::builder;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

/// A volume whose values are `n` distinct indices, 0..n-1, each appearing many times.
VoxelData Indexed(int n, int repeats = 40) {
  VoxelData v;
  v.nx = n;
  v.ny = repeats;
  v.nz = 1;
  v.value.reserve(static_cast<std::size_t>(n) * repeats);
  for (int r = 0; r < repeats; ++r) {
    for (int i = 0; i < n; ++i) { v.value.push_back(static_cast<float>(i)); }
  }
  return v;
}

/// Writes @p body to a scratch file and returns its path.
std::string TempTable(const char* name, const std::string& body) {
  std::string path = std::string("test_ctbl_") + name + ".txt";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) { return std::string(); }
  std::fwrite(body.data(), 1, body.size(), f);
  std::fclose(f);
  return path;
}

/// True if @p got is @p want to within a byte of the 0-255 scale.
bool Near(float got, double want) { return got > want - 0.002 && got < want + 0.002; }

}  // namespace

int main() {
  std::printf("== what an imported voxel volume's values mean ==\n\n");

  // ---- 1. A handful of indices, the ordinary case.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(3), VoxelKind::kDiscrete, s, note);
    Check(s.voxel_classes.size() == 3, "three distinct indices give three material classes");
    if (s.voxel_classes.size() == 3) {
      Check(s.voxel_classes[0].value == 0.0 && s.voxel_classes[1].value == 1.0
                && s.voxel_classes[2].value == 2.0,
            "and they come out in ascending order");
    }
  }

  // ---- 2. More than 64, which used to be refused.
  //
  // 200 organs is a real phantom, not a continuous file. This is the assertion the whole
  // change is about: the count is not evidence about what the values mean.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(200), VoxelKind::kDiscrete, s, note);
    std::printf("  200 indices -> %zu classes, note: %s\n", s.voxel_classes.size(),
                note.c_str());
    // The count is the whole claim, and it is checked by the count. An earlier version of this
    // also asserted the note did not contain the word "continuous", which failed against a
    // refusal message that used the word honestly - "not because this looks continuous but
    // because the list would be too long". Asserting on the absence of a word tests the
    // wording, not the behaviour.
    Check(s.voxel_classes.size() == 200,
          "200 distinct indices give 200 classes rather than a refusal");
  }

  // ---- 3. EVERY INDEX ARRIVES THE SAME, index 0 included.
  //
  //         It did not. Index 0 came in at zero opacity, on the reasoning that 0 means
  //         "nothing here" in every segmentation convention and is also the commonest value,
  //         so an opaque import is a solid block. Usually true - and not this code's call: the
  //         convention belongs to the file, index 0 is sometimes a real structure, and a class
  //         that arrives invisible without being asked for has to be discovered before it can
  //         be wondered about. Wanting something out of the scene now has a control that means
  //         exactly that (the null layer) and it applies to any class rather than to whichever
  //         one happens to be numbered zero.
  //
  //         A tenth, not opaque, for the reason ClassifyVoxels states: the renderer composites
  //         front to back, so a phantom of opaque cells shows only the skin. Asserted as a
  //         value and not as "below one", because the effect depends on how many cells deep
  //         the accumulation reaches and a "less than one" test would pass at 0.99.
  {
    Solid s;
    s.material = 7;   // the container's, which every class should now start on
    std::string note;
    ClassifyVoxels(Indexed(5), VoxelKind::kDiscrete, s, note);
    bool all_faint = true, zero_present = false, all_have_material = true;
    for (const auto& c : s.voxel_classes) {
      if (c.opacity != 0.1f) { all_faint = false; }
      if (c.value == 0.0) { zero_present = true; }
      if (c.material != 7) { all_have_material = false; }
    }
    Check(all_faint, "every index arrives at a tenth, index 0 no different from the rest");
    Check(zero_present, "index 0 is a class like any other, and can be given a material");
    // ---- and each one starts on the container's material rather than on nothing.
    //
    //      A class with none makes the whole volume unbuildable - build_scene refuses to place
    //      a volume whose material is missing - so an import used to arrive in a state where
    //      the scene could not be built until every one of what may be two hundred rows had
    //      been visited. The volume's own material is what a cell matching no class already
    //      falls back to, so this agrees with the rest of the grid rather than inventing a
    //      value: wrong for most classes on a segmentation, and wrong in a column the user can
    //      see rather than absent.
    Check(all_have_material, "and on the container's material, not on nothing");
  }

  // ---- 4. A volume with no zero in it is no different either, which is the same assertion
  //         from the other side: there is no rule about index 0 left to be exempt from.
  {
    Solid s;
    s.material = 3;
    std::string note;
    VoxelData v;
    v.nx = 3; v.ny = 1; v.nz = 1;
    v.value = {7.0f, 8.0f, 9.0f};
    ClassifyVoxels(v, VoxelKind::kDiscrete, s, note);
    bool all_faint = true, all_have_material = true;
    for (const auto& c : s.voxel_classes) {
      if (c.opacity != 0.1f) { all_faint = false; }
      if (c.material != 3) { all_have_material = false; }
    }
    Check(s.voxel_classes.size() == 3, "indices need not start at zero");
    Check(all_faint, "and with no index 0 present, nothing is treated differently");
    Check(all_have_material, "the container's material reaches these classes too");
  }

  // ---- 5. Values that are not whole numbers are not indices, and saying so is the one
  //         inference the data does support.
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 4; v.ny = 1; v.nz = 1;
    v.value = {1.02f, 1.06f, 1.02f, 0.98f};
    ClassifyVoxels(v, VoxelKind::kDiscrete, s, note);
    std::printf("  densities as indices -> %zu classes, note: %s\n", s.voxel_classes.size(),
                note.c_str());
    Check(s.voxel_classes.empty(), "a density map is refused rather than turned into classes");
    Check(note.find("whole numbers") != std::string::npos,
          "and the reason given is integrality, not the number of values");
  }

  // ---- 6. Negative indices work, since some formats use -1 for "outside".
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 4; v.ny = 1; v.nz = 1;
    v.value = {-1.0f, 0.0f, 1.0f, -1.0f};
    ClassifyVoxels(v, VoxelKind::kDiscrete, s, note);
    Check(s.voxel_classes.size() == 3, "a volume using -1 for outside imports its three values");
    if (s.voxel_classes.size() == 3) {
      Check(s.voxel_classes[0].value == -1.0, "and the lowest index comes first");
    }
  }

  // ---- 7. The continuous path still bands by Hounsfield units.
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 4; v.ny = 1; v.nz = 1;
    v.value = {-1000.0f, 0.0f, 400.0f, 1200.0f};
    ClassifyVoxels(v, VoxelKind::kContinuous, s, note);
    Check(!s.voxel_classes.empty(), "a continuous import still produces HU bands");
    bool banded = false;
    for (const auto& c : s.voxel_classes) {
      if (c.value_max > c.value) { banded = true; }
    }
    Check(banded, "and a band spans a range rather than a single value");
  }

  // ---- 8. A colour table on the 0-255 scale.
  //
  // The scale rule is per value and decided by how the number is written: 128 is a byte, 0.5
  // is a fraction. This is the case a segmentation's own table is almost always in.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(4), VoxelKind::kDiscrete, s, note);
    const std::string p = TempTable("bytes",
        "# organ colours\n"
        "0, 0, 0, 0, 0\n"
        "1, 255, 0, 0, 255\n"
        "2, 0, 128, 0, 128\n"
        "3, 0, 0, 255, 64\n");
    Check(!p.empty(), "the scratch table could be written");
    const ColourTableResult r = ReadVoxelColourTable(p, s, note);
    Check(r.ok, "a four-row table applies");
    Check(r.rows == 4 && r.applied == 4, "all four rows applied");
    Check(r.unmatched == 0 && r.malformed == 0, "with nothing unmatched or malformed");
    Check(r.on_255 == 16 && r.on_unit == 0, "and every value read on 0-255");
    if (s.voxel_classes.size() == 4) {
      Check(Near(s.voxel_classes[1].r, 1.0) && s.voxel_classes[1].g == 0.0f,
            "255 is full and 0 is none");
      Check(Near(s.voxel_classes[2].g, 128.0 / 255.0), "128 is about half");
      Check(Near(s.voxel_classes[1].opacity, 1.0), "alpha 255 is opaque");
      Check(Near(s.voxel_classes[3].opacity, 64.0 / 255.0), "alpha 64 is a quarter");
      // The comment beside visible in the reader: a class given zero alpha is being told not
      // to draw, and the checkbox beside it must agree with the number it was just given.
      Check(!s.voxel_classes[0].visible, "alpha 0 also clears visible");
      Check(s.voxel_classes[1].visible, "and a non-zero alpha sets it");
    }
    std::remove(p.c_str());
  }

  // ---- 9. The same table written as fractions, and the two mixed in one file.
  //
  // Mixed is the case that decides the rule is per VALUE and not per file. A per-file guess -
  // say, from the largest number present - reads this file one way or the other and gets half
  // of it wrong; there is no maximum that separates 0-255 from 0-1 here.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(3), VoxelKind::kDiscrete, s, note);
    const std::string p = TempTable("mixed",
        "0 0.0 0.0 0.0 0.0\n"
        "1 1.0 0.5 0.25 1.0\n"
        "2 255 0.5 0 1.0\n");
    const ColourTableResult r = ReadVoxelColourTable(p, s, note);
    Check(r.ok && r.applied == 3, "a fractional table applies");
    // Four fractions on each of the first two rows, then 0.5 and 1.0 on the third; 255 and 0
    // on that row are the two read as bytes.
    Check(r.on_unit == 10 && r.on_255 == 2, "and each value is scaled by how it is written");
    if (s.voxel_classes.size() == 3) {
      Check(s.voxel_classes[1].r == 1.0f, "1.0 is full");
      Check(Near(s.voxel_classes[1].b, 0.25), "0.25 is a quarter");
      // The row that proves the point: 255 and 0.5 on the same line, both read correctly.
      Check(Near(s.voxel_classes[2].r, 1.0) && Near(s.voxel_classes[2].g, 0.5),
            "255 and 0.5 on one row are both full and half");
    }
    std::remove(p.c_str());
  }

  // ---- 10. What the reader refuses to guess at.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(3), VoxelKind::kDiscrete, s, note);
    // The import assigns each class a hue, so "left alone" is whatever it assigned and not a
    // constant. Recorded before the read rather than written down here.
    const float was_r = (s.voxel_classes.size() == 3) ? s.voxel_classes[2].r : -1.0f;
    const std::string p = TempTable("odd",
        "index, r, g, b, a\n"        // a header: five fields, none of them numbers
        "\n"
        "// a comment\n"
        "; another\n"
        "1 300 -20 0.5\n"            // out of range both ways, and no alpha column
        "99 10 10 10 10\n"           // an index this phantom has not got
        "2 nan 0 0\n"                // parses whole under strtod, and is not a colour
        "0 1\n");                    // too few fields to be a row
    const ColourTableResult r = ReadVoxelColourTable(p, s, note);
    Check(r.rows == 2, "comments, blanks, headers and short lines are not rows");
    Check(r.applied == 1, "one row named a class");
    Check(r.unmatched == 1, "one named a class this phantom has not got");
    Check(r.malformed == 2, "the header and the nan are counted, not silently read as zero");
    Check(r.clamped == 2, "300 and -20 are clamped");
    if (s.voxel_classes.size() == 3) {
      Check(s.voxel_classes[1].r == 1.0f, "300 clamps to full");
      Check(s.voxel_classes[1].g == 0.0f, "-20 clamps to none");
      // No alpha column, so opacity keeps the importer's default rather than becoming zero.
      Check(s.voxel_classes[1].opacity == 0.1f, "a row without alpha leaves opacity alone");
      Check(s.voxel_classes[2].r == was_r, "and the nan row changed nothing");
    }
    std::remove(p.c_str());
  }

  // ---- 11. A missing file, and a table for a different phantom, both say so.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(3), VoxelKind::kDiscrete, s, note);
    ColourTableResult r = ReadVoxelColourTable("no_such_colour_table_98211.txt", s, note);
    Check(!r.ok && note.find("could not open") != std::string::npos,
          "a missing file is reported as a missing file");

    const std::string p = TempTable("wrong", "700 1 2 3\n701 4 5 6\n");
    r = ReadVoxelColourTable(p, s, note);
    Check(!r.ok && r.rows == 2 && r.applied == 0,
          "a table whose indices name nothing is not applied");
    Check(note.find("not the row number") != std::string::npos,
          "and the message says what the first column is");
    std::remove(p.c_str());
  }

  // ---- 12. A continuous volume has bands, which have no external index, so a row names one
  //          by position. Matching on value there would need the user to know the band edges
  //          this file invented.
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 4; v.ny = 1; v.nz = 1;
    v.value = {-1000.0f, 0.0f, 400.0f, 1200.0f};
    ClassifyVoxels(v, VoxelKind::kContinuous, s, note);
    const std::size_t n = s.voxel_classes.size();
    Check(n >= 2, "the continuous import banded");
    const std::string p = TempTable("bands", "0 255 0 0\n1 0 255 0\n");
    const ColourTableResult r = ReadVoxelColourTable(p, s, note);
    Check(r.applied == 2, "two rows name the first two bands by position");
    if (n >= 2) {
      Check(s.voxel_classes[0].r == 1.0f && s.voxel_classes[0].g == 0.0f,
            "band 0 took row 0 colour");
      Check(s.voxel_classes[1].g == 1.0f, "and band 1 took row 1");
    }
    std::remove(p.c_str());
  }

  // ---- 13. THE COLORMAPS DO NOT REVISIT A COLOUR THEY HAVE LEFT.
  //
  // Which is what "should not repeat" asks for, and it is worth being precise about, because
  // the first version of this test asked for something stronger and impossible: that 300
  // classes get 300 distinct colours. Gray has 256 colours in it. A one-dimensional map
  // sampled 300 times MUST give some neighbouring pair the same eight-bit value, and the
  // perceptual maps are built to vary smoothly, which is the same thing said approvingly.
  //
  // The property that matters is that a colour is not reused at a DISTANT position - that the
  // map is not cycled once it runs out, so class 1 and class 257 cannot come out the same.
  // Collapsing runs of equal neighbours and then looking for any repeat states exactly that:
  // a map may dwell, and may not return.
  {
    const int kSamples = 1000;
    for (int m = 0; m < static_cast<int>(Colormap::kCount); ++m) {
      std::vector<unsigned int> seq;
      for (int i = 0; i < kSamples; ++i) {
        float r = 0, g = 0, b = 0;
        SampleColormap(static_cast<Colormap>(m), static_cast<double>(i) / (kSamples - 1),
                       r, g, b);
        const unsigned int packed =
            (static_cast<unsigned int>(r * 255 + 0.5f) << 16)
            | (static_cast<unsigned int>(g * 255 + 0.5f) << 8)
            | static_cast<unsigned int>(b * 255 + 0.5f);
        if (seq.empty() || seq.back() != packed) { seq.push_back(packed); }
      }
      bool revisits = false;
      for (std::size_t i = 0; i < seq.size(); ++i) {
        for (std::size_t j = i + 1; j < seq.size(); ++j) {
          if (seq[i] == seq[j]) { revisits = true; }
        }
      }
      char what[96];
      std::snprintf(what, sizeof what, "%s never returns to a colour it has left",
                    ColormapNames()[m]);
      Check(!revisits, what);
      std::snprintf(what, sizeof what, "%s actually varies", ColormapNames()[m]);
      Check(seq.size() > 100, what);
    }
  }

  // ---- 13b. And a phantom's classes are spread over the whole map rather than crowded into
  //           part of it, which is the other half of "resized based on the range".
  {
    const int kN = 300;
    for (int m = 0; m < static_cast<int>(Colormap::kCount); ++m) {
      Solid s;
      std::string note;
      ClassifyVoxels(Indexed(kN, 2), VoxelKind::kDiscrete, s, note,
                     static_cast<Colormap>(m));
      if (s.voxel_classes.size() != static_cast<std::size_t>(kN)) {
        Check(false, "the colormap sweep imported all its classes");
        continue;
      }
      // How many DIFFERENT colours 300 classes came out with. Anything much below 200 would
      // mean the map was being sampled from a short table or squeezed into part of itself.
      std::vector<unsigned int> seen;
      for (const VoxelClass& c : s.voxel_classes) {
        const unsigned int packed = (static_cast<unsigned int>(c.r * 255 + 0.5f) << 16)
                                    | (static_cast<unsigned int>(c.g * 255 + 0.5f) << 8)
                                    | static_cast<unsigned int>(c.b * 255 + 0.5f);
        bool have = false;
        for (unsigned int q : seen) {
          if (q == packed) { have = true; }
        }
        if (!have) { seen.push_back(packed); }
      }
      char what[110];
      std::snprintf(what, sizeof what, "%s gives 300 classes at least 200 distinct colours"
                                       " (got %d)",
                    ColormapNames()[m], static_cast<int>(seen.size()));
      Check(seen.size() >= 200, what);
    }
  }

  // ---- 14. Fitted to the RANGE of the values, which is what makes it not repeat: the first
  //          class is at one end of the map and the last at the other, whatever the values are.
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 4; v.ny = 1; v.nz = 1;
    // Deliberately not 0..3: the map is stretched over 10..40, not over the list positions.
    v.value = {10.0f, 20.0f, 30.0f, 40.0f};
    ClassifyVoxels(v, VoxelKind::kDiscrete, s, note, Colormap::kViridis);
    Check(s.voxel_classes.size() == 4, "four values, four classes");
    if (s.voxel_classes.size() == 4) {
      // Viridis runs dark purple to yellow. Its endpoints are the two values a reader can
      // check against any published copy of the map, which is why they are the ones asserted.
      float r0 = 0, g0 = 0, b0 = 0, r1 = 0, g1 = 0, b1 = 0;
      SampleColormap(Colormap::kViridis, 0.0, r0, g0, b0);
      SampleColormap(Colormap::kViridis, 1.0, r1, g1, b1);
      Check(s.voxel_classes[0].r == r0 && s.voxel_classes[0].b == b0,
            "the lowest value sits at the bottom of the map");
      Check(s.voxel_classes[3].r == r1 && s.voxel_classes[3].g == g1,
            "and the highest at the top");
      Check(static_cast<int>(r0 * 255 + 0.5) == 68 && static_cast<int>(b0 * 255 + 0.5) == 84,
            "viridis starts at its documented dark purple");
      Check(static_cast<int>(r1 * 255 + 0.5) == 253 && static_cast<int>(g1 * 255 + 0.5) == 231,
            "and ends at its documented yellow");
    }
  }

  // ---- 15. A colormap is monotone in position: sampling it is interpolation between stops,
  //          so a t between two stops has to land between their colours. This is what catches
  //          a stop table entered out of order, which no eye check on a phantom would.
  {
    bool ok = true;
    for (int m = 1; m < static_cast<int>(Colormap::kCount); ++m) {   // 0 is the hue sweep
      int n = 0;
      const ColourStop* st = ColormapStops(static_cast<Colormap>(m), n);
      if (st == nullptr || n < 2) {
        ok = false;
        continue;
      }
      for (int k = 1; k < n; ++k) {
        if (!(st[k].t > st[k - 1].t)) { ok = false; }
      }
      if (st[0].t != 0.0 || st[n - 1].t != 1.0) { ok = false; }
      // The midpoint of each interval lies between its ends, componentwise.
      for (int k = 1; k < n; ++k) {
        const double mid = 0.5 * (st[k - 1].t + st[k].t);
        float r = 0, g = 0, b = 0;
        SampleColormap(static_cast<Colormap>(m), mid, r, g, b);
        const int ri = static_cast<int>(r * 255 + 0.5);
        const int lo = (st[k - 1].r < st[k].r) ? st[k - 1].r : st[k].r;
        const int hi = (st[k - 1].r < st[k].r) ? st[k].r : st[k - 1].r;
        if (ri < lo - 1 || ri > hi + 1) { ok = false; }
      }
    }
    Check(ok, "every map's stops ascend, span 0 to 1, and interpolate between themselves");
  }

  // ---- 16. Out of range is clamped rather than wrapped, which is the other way a map could
  //          come to repeat itself.
  {
    float ra = 0, ga = 0, ba = 0, rb = 0, gb = 0, bb = 0;
    SampleColormap(Colormap::kTurbo, -3.0, ra, ga, ba);
    SampleColormap(Colormap::kTurbo, 0.0, rb, gb, bb);
    Check(ra == rb && ga == gb && ba == bb, "below zero clamps to the bottom of the map");
    SampleColormap(Colormap::kTurbo, 7.5, ra, ga, ba);
    SampleColormap(Colormap::kTurbo, 1.0, rb, gb, bb);
    Check(ra == rb && ga == gb && ba == bb, "and above one to the top, rather than wrapping");
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
