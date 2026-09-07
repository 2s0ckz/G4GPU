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
  {
    Solid s;
    std::string note;
    ClassifyVoxels(Indexed(5), VoxelKind::kDiscrete, s, note);
    bool zero_hidden = false, others_opaque = true;
    for (const auto& c : s.voxel_classes) {
      if (c.value == 0.0) {
        zero_hidden = (c.opacity == 0.0f);
      } else if (c.opacity != 1.0f) {
        others_opaque = false;
      }
    }
    Check(zero_hidden, "index 0 is imported with zero opacity");
    Check(others_opaque, "and every other index is left opaque");
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
    bool all_opaque = true;
    for (const auto& c : s.voxel_classes) {
      if (c.opacity != 1.0f) { all_opaque = false; }
    }
    Check(s.voxel_classes.size() == 3, "indices need not start at zero");
    Check(all_opaque, "and with no index 0 present, nothing is hidden");
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

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
