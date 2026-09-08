// A voxel class can be on its own layer, and the space it wins is the space it transports.
//
// WHY THIS EXISTS
//
// The layer rule gives shared space to the higher layer, and a voxel volume had one layer for
// all of it. So a phantom overlapping a seat, a couch, a helmet or an implant was either
// always on top - its AIR cells winning over solid aluminium - or always underneath, with its
// bone losing to the same. Neither is what the geometry means. Per-class layers are the third
// answer: one volume that outranks its neighbour where it is bone and loses where it is air.
//
// That makes a volume's priority a function of the POINT, which is a change to the rule the
// stepper applies rather than to anything cosmetic. Three things in the navigator had to
// follow it, and each can be wrong on its own:
//
//   * locate() has to ask the class at the point rather than read one number off the volume;
//   * a step inside a grid has to end where the class LAYER changes, not only where the
//     material does - two classes may share a material and rank differently, and a step that
//     crossed both would have the volumes overlapping the far half of it go unseen;
//   * and a volume looking AT such a grid cannot use "where does its box start" as "where
//     does it start outranking me", because the answer is a cell boundary inside it.
//
// The first two are checked exactly, with no beam, in tests/test_voxels.cu. This file is the
// one that puts a beam through it, and what it checks is an EQUIVALENCE:
//
//   scene A  a grid whose class 0 is air on the volume's layer and whose class 1 is water on a
//            higher one, with a box of tissue covering the whole grid on a layer in between.
//            Class 0 loses its cells to the box; class 1 keeps its own.
//   scene B  the same grid with class 0's MATERIAL set to tissue, no box, and no layers.
//
// Every point of the two scenes is the same material, by two entirely different mechanisms -
// one that needs all of the machinery above and one that needs none of it. So they have to
// deposit the same energy, and every cell has to report the same material. If the per-class
// path is wrong anywhere, B is what A should have been.
//
// The last check is the one that keeps the rest honest: scene A must actually have per-class
// layers in its flattened pool and scene B must not. Both scenes reading as "no per-class
// layers" would make every check above pass while testing nothing at all, and that is exactly
// what a bug in the model-to-scene step would look like.
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
  std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
  if (!ok) { ++g_fails; }
}

constexpr double kHalf = 100.0;   ///< mm; 200 mm along the beam
constexpr int kNz = 20;           ///< cells along z, so 10 mm cells

/// Model material indices, with spares between the placed ones so model order and device
/// order cannot coincide by luck - the same trap tests/test_voxel_materials.cu sets.
enum : int { kAir = 0, kSpareBone = 1, kWater = 2, kSpareMuscle = 3, kTissue = 4 };

const char* const kNist[] = {"G4_AIR", "G4_BONE_COMPACT_ICRU", "G4_WATER",
                             "G4_MUSCLE_SKELETAL_ICRP", "G4_TISSUE_SOFT_ICRP"};

/// @param by_layer  true for scene A - class 0 on the volume's layer, class 1 above the box,
///                  and a box of tissue over the grid. False for scene B, which says the same
///                  thing with class 0's material and no box at all.
Model MakeModel(bool by_layer, const char* particle, double energy) {
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
  tgt.name = "Phantom";
  tgt.layer = 1;
  tgt.material = kAir;
  tgt.p[0] = tgt.p[1] = tgt.p[2] = kHalf;
  tgt.shape = Shape::kVoxelGrid;
  tgt.p[3] = 1;
  tgt.p[4] = 1;
  tgt.p[5] = kNz;
  tgt.voxel_kind = VoxelKind::kDiscrete;
  tgt.voxel_meaning = VoxelMeaning::kIndices;
  tgt.voxel_values.assign(static_cast<std::size_t>(kNz), 0.0f);
  for (int k = 0; k < kNz; ++k) {
    tgt.voxel_values[static_cast<std::size_t>(k)] = static_cast<float>(k % 2);
  }
  {
    VoxelClass c0;
    c0.value = 0;
    // The whole difference between the two scenes: in A this class is air and loses its space
    // to the box; in B it IS the box's material and there is no box.
    c0.material = by_layer ? kAir : kTissue;
    c0.layer = kInheritLayer;
    tgt.voxel_classes.push_back(c0);
    VoxelClass c1;
    c1.value = 1;
    c1.material = kWater;
    // Above the box in A, so it keeps its own cells. Inheriting in B, where nothing overlaps.
    c1.layer = by_layer ? 5 : kInheritLayer;
    tgt.voxel_classes.push_back(c1);
  }
  m.solids.push_back(tgt);

  if (by_layer) {
    Solid cover;
    cover.name = "Cover";
    cover.shape = Shape::kBox;
    // EXACTLY the grid's extent, so the only difference between the scenes is the mechanism
    // and not the shape of anything.
    cover.p[0] = cover.p[1] = cover.p[2] = kHalf;
    cover.material = kTissue;
    cover.layer = 3;   // above class 0, below class 1
    m.solids.push_back(cover);
  }

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
  sc.solids.push_back(1);                    // the phantom
  if (by_layer) { sc.solids.push_back(2); }  // and the box, which owns half of its cells
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

/// Every scorer's total, over the DE-DUPLICATED list.
///
/// Not by walking Detectors(): SetSensitiveDetector registers its detector once per volume, so
/// a scorer attached to two volumes appears twice there and summing it twice reports double
/// the energy - which is what this file's equivalence check first reported, exactly 2.0000x,
/// for a scorer that covered a phantom and the box overlapping it. G4SDManager::Scorers() is
/// the list AssignIndices built, one entry per slot.
double ScorerTotal() {
  double t = 0;
  for (G4VPrimitiveScorer* ps : G4SDManager::GetSDMpointer()->Scorers()) { t += ps->total; }
  return t;
}

/// What the scene says the material is at each cell centre, and whether it carries per-class
/// layers at all. Read from the FLATTENED scene through locate() and material_at(), which are
/// the functions the stepper calls.
struct Probe {
  std::vector<int> material;   ///< device material index per cell centre
  std::vector<int> owner;      ///< volume index that owns each cell centre
  bool has_class_layers = false;
};

double Run(bool by_layer, const char* particle, double energy, int events, Probe* probe) {
  ClearRegistries();
  Model m = MakeModel(by_layer, particle, energy);
  auto* rm = new G4RunManager;
  rm->SetUserInitialization(new ModelDetector(&m));
  rm->SetUserAction(new ModelPrimary(&m));
  rm->Initialize();

  if (probe != nullptr) {
    const auto& scene = rm->GetScene();
    geom::Geometry<G4double> g{};
    g.volumes = scene.volumes.data();
    g.n_volumes = static_cast<int>(scene.volumes.size());
    g.world = scene.world;
    g.store = scene.pool.store();
    g.voxels = scene.pool.voxel_store();
    probe->has_class_layers = scene.pool.any_class_layers;
    probe->material.clear();
    probe->owner.clear();
    const double cell = 2 * kHalf / kNz;
    for (int k = 0; k < kNz; ++k) {
      const g4gpu::Vec3<G4double> p{0, 0, -kHalf + (k + 0.5) * cell};
      const int own = geom::locate(g, p);
      probe->owner.push_back(own);
      probe->material.push_back((own >= 0) ? geom::material_at(g, own, p) : -1);
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
  std::printf("== a voxel class on its own layer transports as what it wins ==\n");

  struct Case {
    const char* particle;
    double energy;
  };
  const Case cases[] = {{"gamma", 1.0}, {"e-", 6.0}, {"proton", 80.0}};

  for (const Case& c : cases) {
    Probe pa, pb;
    const double a = Run(true, c.particle, c.energy, events, &pa);
    const double b = Run(false, c.particle, c.energy, events, &pb);

    std::printf("\n  %s %g MeV\n", c.particle, c.energy);

    // The check that keeps the others honest. Both scenes reading "no per-class layers" would
    // make everything below pass while testing nothing.
    Check(pa.has_class_layers,
          "the scene built with per-class layers actually carries them");
    Check(!pb.has_class_layers,
          "and the one that says the same thing by material carries none");

    // Every cell the same material, by two different mechanisms.
    int diff = 0;
    for (std::size_t k = 0; k < pa.material.size() && k < pb.material.size(); ++k) {
      if (pa.material[k] != pb.material[k]) { ++diff; }
    }
    std::printf("    materials per cell: ");
    for (std::size_t k = 0; k < pa.material.size() && k < 8; ++k) {
      std::printf("%d/%d ", pa.material[k], pb.material[k]);
    }
    std::printf("(owner in A: ");
    for (std::size_t k = 0; k < pa.owner.size() && k < 8; ++k) {
      std::printf("%d ", pa.owner[k]);
    }
    std::printf(")\n");
    Check(diff == 0, "every cell centre is the same material in both scenes");

    // And that the two mechanisms really are different: in A the box owns the even cells.
    int owned_by_cover = 0;
    for (std::size_t k = 0; k < pa.owner.size(); ++k) {
      if (pa.owner[k] == 2) { ++owned_by_cover; }
    }
    Check(owned_by_cover == kNz / 2,
          "in A the covering box owns exactly the cells whose class it outranks");

    // The energy. Statistical: the two scenes have different volume boundaries in the same
    // places, so the step sequences are not identical even at one seed.
    const double rel = (b > 0) ? std::fabs(a - b) / b : std::fabs(a - b);
    std::printf("    deposit per event: %.6g MeV by layer, %.6g MeV by material (%.3g%%)\n",
                a, b, 100 * rel);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s deposits the same either way (within 2%%)",
                  c.particle);
    Check(rel < 0.02, buf);
  }

  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "ALL PASS", g_fails);
  return g_fails ? 1 : 0;
}
