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

  // ---- 3. Index 0 is hidden, because 0 means "nothing here" in every segmentation
  //         convention and is also the commonest value, so an opaque import is a solid block.
  //
  //         The rest are faint rather than opaque, for the reason ClassifyVoxels states: the
  //         renderer composites front to back, so a phantom of opaque cells shows only the
  //         skin. This asserts the value, not just that it is below one, because the whole
  //         effect depends on how many cells deep the accumulation reaches - and a "less than
  //         one" test would pass at 0.99, which shows the skin.
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(5), VoxelKind::kDiscrete, s, note);
    bool zero_hidden = false, others_faint = true;
    for (const auto& c : s.voxel_classes) {
      if (c.value == 0.0) {
        zero_hidden = (c.opacity == 0.0f);
      } else if (c.opacity != 0.1f) {
        others_faint = false;
      }
    }
    Check(zero_hidden, "index 0 is imported with zero opacity");
    Check(others_faint, "and every other index at a tenth, so the interior reads through");
    // Hidden, not removed: it still has to be assignable to a material.
    bool zero_present = false;
    for (const auto& c : s.voxel_classes) {
      if (c.value == 0.0) { zero_present = true; }
    }
    Check(zero_present, "index 0 is still a class, so it can still be given a material");
  }

  // ---- 4. A volume with no zero is untouched by that rule.
  {
    Solid s;
    std::string note;
    VoxelData v;
    v.nx = 3; v.ny = 1; v.nz = 1;
    v.value = {7.0f, 8.0f, 9.0f};
    ClassifyVoxels(v, VoxelKind::kDiscrete, s, note);
    bool all_faint = true;
    for (const auto& c : s.voxel_classes) {
      if (c.opacity != 0.1f) { all_faint = false; }
    }
    Check(s.voxel_classes.size() == 3, "indices need not start at zero");
    Check(all_faint, "and with no index 0 present, nothing is fully hidden");
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

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
