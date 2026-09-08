// A voxel cell's material is the one the user assigned to its class.
//
// WHY THIS EXISTS
//
// Reported as "gammas pass straight through the phantom even at low energy, and electrons and
// positrons cannot penetrate a single non-air voxel at any energy" - two opposite symptoms,
// one cause.
//
// The builder fills a grid's cells during Construct(), from `Model::materials` indices,
// because that is the only numbering it has: DEVICE material indices are handed out later, in
// G4Flatten, from the materials of the PLACED volumes. The transport then read those numbers
// as device indices. Two things were wrong at once:
//
//   1. NO TRANSLATION. Model order and device order coincide only when every model material,
//      in order, is also some placed volume's material. One material in the list that the
//      user has not placed slides the numbering by one.
//
//   2. A MATERIAL USED ONLY BY VOXEL CLASSES WAS NEVER BUILT. G4Flatten walked placements and
//      took each logical volume's material; nothing walked voxel classes. For a segmented
//      phantom that is the normal case - the volume has one material and its two hundred
//      classes have others - so those materials kept device_index -1 and never entered the
//      table at all.
//
// The cell index was therefore out of range, and the kernels read past the end of the material
// table. Garbage cross-sections gave a gamma an enormous mean free path, so it crossed 200 mm
// of "water" depositing nothing; garbage stopping power made a 100 MeV electron dump 99.996 of
// its 100 MeV in three steps. Both symptoms, from one wrong index.
//
// WHAT IS ASSERTED, AND IN WHICH ORDER
//
// The end-to-end check - does a voxel phantom deposit what an identical box deposits - is
// statistical and it is last. Before it come two exact ones, because an exact check says which
// number is wrong rather than that some number is:
//
//   1. every material a voxel class names has a device index at all;
//   2. every cell holds the DEVICE index of the material its class was assigned.
//
// The second is the invariant that was broken, stated directly. It needs no beam.
//
// The grid is deliberately given a material NO PLACED VOLUME USES, and the model list
// deliberately has spare materials in it. Both are what a real phantom looks like and both are
// what the old code got wrong; a test built from Air and Water alone passes either way, which
// is why this went unnoticed - the builder's own selftest phantom is exactly that.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "builder/build_scene.hh"
#include "builder/model.hh"

using namespace g4gpu;
using namespace g4gpu::builder;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

constexpr double kHalf = 100.0;   ///< mm; 200 mm of material along the beam
constexpr int kNz = 40;

/// Model material indices. The spares sit BETWEEN the ones that get placed, so model order
/// and device order cannot coincide by luck.
enum : int { kAir = 0, kSpareBone = 1, kWater = 2, kSpareMuscle = 3, kTissue = 4 };

const char* const kNist[] = {"G4_AIR", "G4_BONE_COMPACT_ICRU", "G4_WATER",
                             "G4_MUSCLE_SKELETAL_ICRP", "G4_TISSUE_SOFT_ICRP"};

/// @param voxels    a voxel grid rather than a box
/// @param cell_mat  model material index for the cells, and for the box, so the two are
///                  comparable. kTissue is used by nothing else, which is the case under test.
Model MakeModel(bool voxels, int cell_mat, const char* particle, double energy) {
  Model m;
  m.elements.clear();
  m.materials.clear();
  m.solids.clear();
  m.sources.clear();
  m.scorers.clear();

  for (const char* n : kNist) {
    Material mm;
    mm.name = n;
    mm.nist_name = n;
    m.materials.push_back(mm);
  }

  Solid world;
  world.name = "World";
  world.shape = Shape::kBox;
  world.p[0] = world.p[1] = world.p[2] = 400;
  world.material = kAir;
  world.layer = 0;
  m.solids.push_back(world);

  Solid tgt;
  tgt.name = "Target";
  tgt.layer = 1;
  tgt.material = cell_mat;
  tgt.p[0] = tgt.p[1] = tgt.p[2] = kHalf;
  if (voxels) {
    tgt.shape = Shape::kVoxelGrid;
    tgt.p[3] = 1;
    tgt.p[4] = 1;
    tgt.p[5] = kNz;
    tgt.voxel_kind = VoxelKind::kDiscrete;
    tgt.voxel_meaning = VoxelMeaning::kIndices;
    tgt.voxel_values.assign(static_cast<std::size_t>(kNz), 1.0f);
    VoxelClass c0;         // class 0 is air, as an imported phantom has it
    c0.value = 0;
    c0.material = kAir;
    tgt.voxel_classes.push_back(c0);
    VoxelClass c1;
    c1.value = 1;
    c1.material = cell_mat;
    tgt.voxel_classes.push_back(c1);
  } else {
    tgt.shape = Shape::kBox;
  }
  m.solids.push_back(tgt);

  builder::Source src;
  src.name = "Beam";
  src.kind = SourceKind::kBeam;
  src.particle = particle;
  src.energy_MeV = energy;
  src.pos[2] = -kHalf - 10.0;
  src.dir[2] = 1;
  src.half_x = 0.01;
  src.half_y = 0.01;
  m.sources.push_back(src);

  Scorer sc;
  sc.name = "edep";
  sc.quantity = ScoreQuantity::kEnergyDeposit;
  sc.solids.push_back(1);   // the Target
  m.scorers.push_back(sc);
  return m;
}

void ClearRegistries() {
  G4PVPlacement::Registry().clear();
  G4LogicalVolume::Registry().clear();
  G4Material::Registry().clear();
  G4Element::Registry().clear();
  G4SDManager::GetSDMpointer()->Reset();
}

G4Material* FindByName(const char* nist) {
  for (G4Material* m : G4Material::Registry()) {
    if (m->GetName() == nist) { return m; }
  }
  return nullptr;
}

double ScorerTotal() {
  double t = 0;
  for (G4VSensitiveDetector* sd : G4SDManager::GetSDMpointer()->Detectors()) {
    for (G4VPrimitiveScorer* ps : sd->GetScorers()) { t += ps->total; }
  }
  return t;
}

/// Runs one geometry and returns the deposit per event in MeV.
double Run(bool voxels, int cell_mat, const char* particle, double energy, int events,
           bool check_indices) {
  ClearRegistries();
  Model m = MakeModel(voxels, cell_mat, particle, energy);
  auto* rm = new G4RunManager;
  rm->SetUserInitialization(new ModelDetector(&m));
  rm->SetUserAction(new ModelPrimary(&m));
  rm->Initialize();

  if (check_indices) {
    // ---- 1. Every material a class names has a device index.
    //
    // Before the fix this failed for exactly the materials that matter: the ones no placed
    // volume uses. -1 means it was never built into the table the kernels read.
    for (const VoxelClass& c : m.solids[1].voxel_classes) {
      G4Material* mat = FindByName(kNist[c.material]);
      char buf[160];
      std::snprintf(buf, sizeof buf,
                    "class value %g names %s, which was built for the device", c.value,
                    kNist[c.material]);
      Check(mat != nullptr && mat->device_index >= 0, buf);
    }

    // ---- 2. Every cell holds the DEVICE index of its class's material.
    //
    // The invariant that was broken, stated with no beam and no statistics. The cells here
    // are all class 1, so every one of them must carry cell_mat's device index - which is NOT
    // cell_mat, because the model list has materials in it that nothing places.
    G4LogicalVolume* lv = nullptr;
    for (G4PVPlacement* p : G4PVPlacement::Registry()) {
      if (p->GetName() == "Target") { lv = p->GetLogicalVolume(); }
    }
    auto* grid = (lv != nullptr) ? dynamic_cast<G4VoxelGrid*>(lv->GetSolid()) : nullptr;
    Check(grid != nullptr, "the target is a voxel grid");
    G4Material* want = FindByName(kNist[cell_mat]);
    if (grid != nullptr && want != nullptr) {
      // GetCell reports what the builder wrote - model indices. Build() is what translates,
      // so the check is on the grid's own material table, which is what Build() uses.
      const std::vector<G4Material*>& tab = grid->CellMaterials();
      Check(!tab.empty(),
            "the grid says what its cell numbers mean, so Build can translate them");
      bool all = !tab.empty();
      for (int k = 0; k < kNz && all; ++k) {
        const short v = grid->GetCell(0, 0, k);
        if (v < 0 || static_cast<std::size_t>(v) >= tab.size()) { all = false; break; }
        if (tab[static_cast<std::size_t>(v)] != want) { all = false; }
      }
      Check(all, "every cell resolves through that table to the class's own material");

      // And the translation itself: what would actually go to the device.
      const int dev = want->device_index;
      std::printf("  %s is model index %d and device index %d\n", kNist[cell_mat], cell_mat,
                  dev);
      Check(dev != cell_mat || cell_mat == 0,
            "the two numberings differ here, which is the case the old code got wrong");
    }
  }

  rm->BeamOn(events);
  const double per_event = ScorerTotal() / MeV / events;
  delete rm;
  return per_event;
}

}  // namespace

int main(int argc, char** argv) {
  const int events = (argc > 1) ? std::atoi(argv[1]) : 20000;
  std::printf("== a voxel cell's material is its class's material ==\n\n");

  // ---- The exact checks, on a grid whose cells use a material nothing else does.
  std::printf("exact checks, cells of %s:\n", kNist[kTissue]);
  Run(true, kTissue, "gamma", 6.0, 200, true);

  // ---- The end-to-end check: the same material as a box and as a grid.
  //
  // Two per cent, which is wide for 20000 events and deliberately so: this is the backstop,
  // and what it has to separate is not two close numbers but a right answer from a zero. The
  // failures it exists to catch were x0.00 for a gamma and x2.24 for an electron.
  struct Beam {
    const char* particle;
    double energy;
  };
  const Beam kBeams[] = {{"gamma", 6.0}, {"gamma", 0.1}, {"e-", 100.0}, {"e-", 10.0},
                         {"e+", 20.0}};
  std::printf("\n  %-14s %14s %14s %8s\n", "beam", "box MeV/evt", "grid MeV/evt", "ratio");
  for (const Beam& b : kBeams) {
    const double box = Run(false, kTissue, b.particle, b.energy, events, false);
    const double grid = Run(true, kTissue, b.particle, b.energy, events, false);
    const double ratio = (box > 0) ? grid / box : 0.0;
    char label[64];
    std::snprintf(label, sizeof label, "%s %g MeV", b.particle, b.energy);
    std::printf("  %-14s %14.6f %14.6f %8.4f\n", label, box, grid, ratio);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s deposits the same in a grid as in a box (%.4f)", label,
                  ratio);
    Check(box > 0 && ratio > 0.98 && ratio < 1.02, buf);
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
