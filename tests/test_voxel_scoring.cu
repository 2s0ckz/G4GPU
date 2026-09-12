// Per-voxel scoring: that a deposit lands in the cell it happened in.
//
// WHY THIS EXISTS
//
// The check that was there asserts the per-cell sums add up to the volume total, and passes
// whether or not any deposit landed in the right cell - a step crossing forty cells and dumping
// everything into the last one sums identically. `voxel_step`'s own comment records that a
// previous version of exactly that bug zeroed every score in the scene, so the failure is not
// hypothetical and the check that survived it cannot see it.
//
// Two assertions, and the first is the sharper one:
//
//   1. EVERY DEPOSITING STEP LIES WITHIN ONE CELL. That is the property the whole mechanism
//      rests on - `every_cell` in voxel_step, switched on by score_per_voxel in G4Flatten - and
//      it is exact, so it needs no statistics. A step whose two ends are in different cells has
//      a deposit with no single cell to belong to, and attributing it is then a guess.
//
//   2. BINNED BY HAND, THE STEPS REPRODUCE THE SCORER CELL BY CELL. The scorer accumulates on
//      the device as the steps happen; this bins the same steps on the host afterwards from
//      their own positions. Two routes to the same 64 numbers, so agreement is evidence rather
//      than tautology - and a deposit in the wrong cell moves one number up and another down,
//      which a total cannot see and a per-cell comparison cannot miss.
//
// A CUSTOM HOOK, AND WHY THE STOCK ONE WOULD NOT DO
//
// The first version of this used StepTap, and StepRecord keeps a step's POST-step position and
// its length - not where it started. Both assertions need the pre-step point: the kernel picks
// the cell from `p.pos` BEFORE the step runs (see run_step_gamma), so binning by the post-step
// point disagrees with the scorer at every boundary, and "the step is longer than a cell" is
// not the same claim as "the step crossed a cell boundary" - a diagonal step can be longer than
// a cell and stay inside it. Both of those made the first version fail against a port that was
// behaving correctly, which is worth writing down: a new check that fails is evidence about the
// check until it is evidence about the code.
//
// So the hook records both ends. That is what the step hook is for, and it costs this file its
// own instantiation of the transport - see the note in transport_run_impl.cuh. Since P8e that
// instantiation is the ENGINE only: the kernels for this hook are compiled one per translation
// unit by build_hook_engine.bat, because eighteen of them in this file is the unit ptxas
// refuses (docs/RISK.md V65).
//
// The geometry is a stack of 64 identical water slabs, which is the worst case on purpose:
// nothing about the material changes from cell to cell, so the only thing that can end a step
// at a boundary is the per-voxel rule. A phantom of mixed materials would pass assertion 1 by
// accident wherever the material happened to change.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

// The hook and the record it writes are include/CellTap.hh, because since P8e the kernels for
// this hook type are compiled one per translation unit and each of those units includes it.
#include "CellTap.hh"

namespace {

constexpr int kNz = 64;             ///< cells along the beam
constexpr double kHalfZ = 40.0;     ///< mm, half the phantom's depth
constexpr double kHalfXY = 30.0;    ///< mm

}  // namespace

// The engine for this hook, DECLARING its kernels rather than defining them. This file used to
// define all eighteen - "it costs this file its own instantiation of the transport", as the
// note above still says - and with the Urban ion branch live that stopped compiling, exactly as
// tests/test_custom_hook.cu did and for the same reason. build_hook_engine.bat compiles them
// one per unit into out\hook_cell.lib; hook_kernels.cuh is the declarations that keep them out
// of here, and must follow transport_run_impl.cuh, whose macros they are written against.
#define G4STEP_HOOK CellTap
#include "g4/G4NistManager.hh"
#include "g4/G4RunManager.hh"
#include "g4/G4SDManager.hh"
#include "g4/G4Solids.hh"
#include "g4/G4UserActions.hh"
#include "host/transport_run_impl.cuh"

#include "hook_kernels.cuh"  // generated beside out\hook_cell.lib by build_hook_engine.bat

namespace g4gpu::host {
template class TransportEngine<double, CellTap>;
}  // namespace g4gpu::host

using namespace g4gpu;

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++g_fails;
  }
}

/// Cell index along z for a point, or -1 outside. The phantom sits at the origin unrotated, so
/// global and local coordinates are the same - deliberately, because a transform here would be
/// a second place for this arithmetic to be wrong.
int CellOfZ(double z) {
  if (z < -kHalfZ || z > kHalfZ) { return -1; }
  const double cell = 2.0 * kHalfZ / kNz;
  int k = static_cast<int>((z + kHalfZ) / cell);
  if (k < 0) { k = 0; }
  if (k >= kNz) { k = kNz - 1; }
  return k;
}

class Phantom : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

    auto* world_s = new G4Box("World", 200, 200, 200);
    auto* world_l = new G4LogicalVolume(world_s, air, "World");
    world_ = new G4PVPlacement(nullptr, G4ThreeVector(), world_l, "World", nullptr, false, 0);

    auto* grid = new G4VoxelGrid("Phantom", kHalfXY, kHalfXY, kHalfZ, 1, 1, kNz);
    for (int k = 0; k < kNz; ++k) { grid->SetCell(0, 0, k, water->device_index); }
    auto* grid_l = new G4LogicalVolume(grid, water, "Phantom");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), grid_l, "Phantom", world_l, false, 0);
    grid_lv_ = grid_l;
    return world_;
  }

  void ConstructSDandField() override {
    // G4PSEnergyDeposit3D is the per-voxel primitive; IsPerVoxel() returning true is what
    // G4Flatten reads to switch every_cell on for this volume.
    scorer_ = new G4PSEnergyDeposit3D("cells");
    auto* det = new G4MultiFunctionalDetector("PhantomSD");
    det->RegisterPrimitive(scorer_);
    SetSensitiveDetector(grid_lv_, det);
  }

  G4PSEnergyDeposit3D* Scorer() const { return scorer_; }

 private:
  G4VPhysicalVolume* world_ = nullptr;
  G4LogicalVolume* grid_lv_ = nullptr;
  G4PSEnergyDeposit3D* scorer_ = nullptr;
};

/// A pencil beam of 6 MeV gammas down the axis, entering the front face.
class Beam : public G4VUserPrimaryGeneratorAction {
 public:
  Beam() {
    gun_ = new G4ParticleGun(1);
    gun_->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("gamma"));
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun_->SetParticleEnergy(6.0 * MeV);
    gun_->SetParticlePosition(G4ThreeVector(0, 0, -150));
    if (G4RunManager::Instance() != nullptr) { G4RunManager::Instance()->SetGun(gun_); }
  }
  ~Beam() override { delete gun_; }
  void GeneratePrimaries(G4Event* e) override { gun_->GeneratePrimaryVertex(e); }

 private:
  G4ParticleGun* gun_ = nullptr;
};

}  // namespace

int main() {
  std::printf("== a deposit lands in the cell it happened in ==\n\n");

  auto* rm = new G4RunManager;
  auto* det = new Phantom;
  rm->SetUserInitialization(det);
  rm->SetUserAction(new Beam);
  rm->SetCutValue(0.7 * mm);
  rm->Initialize();

  // Small, and generously sized, because this asserts nothing was dropped: a truncated sample
  // makes the host binning short and the per-cell comparison meaningless. Every cell boundary
  // ends a step here, so a track crossing the phantom is 64 steps before any secondary - the
  // step count per event is far higher than a B1-shaped guess would suggest, and the first
  // version of this file was sized by such a guess and overflowed.
  const int kEvents = 2000;
  const int capacity = 4 << 20;

  CellRec* d_rec = nullptr;
  int* d_ctl = nullptr;
  if (cudaMalloc(&d_rec, sizeof(CellRec) * static_cast<size_t>(capacity)) != cudaSuccess
      || cudaMalloc(&d_ctl, sizeof(int) * 2) != cudaSuccess) {
    std::printf("  FAIL: could not allocate the tap\n");
    return 1;
  }
  cudaMemset(d_ctl, 0, sizeof(int) * 2);
  rm->SetStepHook(CellTap(d_rec, d_ctl, capacity));

  rm->BeamOn(kEvents);

  int ctl[2] = {0, 0};
  cudaMemcpy(ctl, d_ctl, sizeof(ctl), cudaMemcpyDeviceToHost);
  const int n_rec = ctl[0] < capacity ? ctl[0] : capacity;
  std::printf("  %d steps recorded in the phantom over %d events (%.0f per event), %d dropped\n",
              n_rec, kEvents, double(n_rec) / kEvents, ctl[1]);
  Check(ctl[1] == 0, "the tap dropped nothing, so the host binning is complete");
  Check(n_rec > 1000, "the run produced steps to check");

  std::vector<CellRec> rec(static_cast<size_t>(n_rec));
  if (!rec.empty()) {
    cudaMemcpy(rec.data(), d_rec, sizeof(CellRec) * rec.size(), cudaMemcpyDeviceToHost);
  }

  // ---- 1. Every depositing step ENDS AT a cell boundary or stays inside one cell.
  //
  // Not "both ends in the same cell", which is what this asserted first and which fails against
  // a port that is behaving correctly. A step that runs to a boundary has its post-step point
  // nudged kPushDistance PAST it, deliberately, so the next locate() resolves in the new cell
  // rather than landing exactly on the face - so most steps here legitimately report two
  // different cells. Measured, that overshoot is 1.000e-07 mm on every one of them, which is
  // kPushDistance for double exactly.
  //
  // So the property is: a step may end in the next cell only by that nudge. A step whose post
  // point is a real distance past the boundary travelled THROUGH it, and its deposit then has
  // no single cell to belong to - which is the failure this exists to catch.
  const double kTol = 10.0 * geom::kPushDistance<G4double>();
  const double kCell = 2.0 * kHalfZ / kNz;
  long long crossed = 0, at_boundary = 0, inside = 0;
  int worst_span = 0;
  double worst_over = 0;
  for (const auto& r : rec) {
    if (r.edep == 0.0) { continue; }
    const int a = CellOfZ(r.pre_z);
    const int b = CellOfZ(r.post_z);
    if (a < 0) { continue; }
    ++inside;
    if (b < 0 || b == a) { continue; }
    const double edge = -kHalfZ + kCell * ((b > a) ? b : a);
    const double over = std::fabs(r.post_z - edge);
    if (over > worst_over) { worst_over = over; }
    if (over <= kTol) {
      ++at_boundary;
    } else {
      ++crossed;
      const int d = (b > a) ? (b - a) : (a - b);
      if (d > worst_span) { worst_span = d; }
    }
  }
  std::printf("  %lld depositing steps: %lld ended exactly at a cell boundary, %lld went\n"
              "  through one; worst overshoot %.3e mm against a %.3f mm cell\n",
              inside, at_boundary, crossed, worst_over, kCell);
  Check(inside > 500, "there are depositing steps inside the phantom to check");
  Check(at_boundary > 100,
        "steps do end at cell boundaries, so the per-voxel rule is actually in force");
  Check(crossed == 0, "and no depositing step travelled through a boundary");
  if (crossed != 0) {
    std::printf("        the worst spans %d cells. Each voxel boundary is supposed to end a\n"
                "        step - voxel_step's every_cell, switched on by score_per_voxel in\n"
                "        G4Flatten. A deposit spanning cells has no single cell to go in.\n",
                worst_span);
  }

  // ---- 2. Binned by hand from the PRE-step point, which is the cell the kernel charges.
  std::vector<double> host(kNz, 0.0);
  for (const auto& r : rec) {
    if (r.edep == 0.0) { continue; }
    const int k = CellOfZ(r.pre_z);
    if (k >= 0) { host[static_cast<size_t>(k)] += r.edep; }
  }

  const std::vector<G4double>& dev = det->Scorer()->cells;
  Check(dev.size() >= static_cast<size_t>(kNz), "the scorer reported a cell array");
  if (dev.size() >= static_cast<size_t>(kNz)) {
    double worst = 0, dev_total = 0, host_total = 0;
    int worst_k = -1, hit = 0;
    for (int k = 0; k < kNz; ++k) {
      const double d = dev[static_cast<size_t>(k)], h = host[static_cast<size_t>(k)];
      dev_total += d;
      host_total += h;
      if (d > 0) { ++hit; }
      const double rel = std::fabs(d - h) / ((d > 0) ? d : 1.0);
      if (rel > worst) {
        worst = rel;
        worst_k = k;
      }
    }
    std::printf("  %d of %d cells took energy; device %.6f MeV, host %.6f MeV\n", hit, kNz,
                dev_total, host_total);
    std::printf("  worst per-cell disagreement %.3e at cell %d\n", worst, worst_k);
    Check(hit > kNz / 4, "a 6 MeV beam deposits over a good part of the depth, not one slab");
    Check(worst < 1e-9, "every cell's deposit matches the steps that made it, cell by cell");
    if (worst >= 1e-9) {
      std::printf("        A deposit in the wrong cell moves one of these up and another down,\n"
                  "        which the sum over cells cannot see - and that sum is the only\n"
                  "        thing that used to be checked.\n");
    }
    // The weaker, older claim, kept as an anchor: if this fails too then something is lost
    // rather than misplaced, which is a different fault with a different cause.
    const double rel = (dev_total > 0) ? std::fabs(dev_total - host_total) / dev_total : 0.0;
    Check(rel < 1e-12, "and the totals agree, so nothing is lost as well as misplaced");
  }

  cudaFree(d_rec);
  cudaFree(d_ctl);
  std::printf("\n%s (%d failures)\n", g_fails ? "FAILED" : "PASSED", g_fails);
  return g_fails ? 1 : 0;
}
