// P3's nuclear level data on the device: PhotonEvaporation5.7, read once and uploaded flat.
//
// `data::LevelTable` was built for this from the start - four raw pointers and three counts,
// with `LevelTableStorage` owning the host vectors it points into - so what was missing was not
// a data structure but a `cudaMalloc`. This file is that, plus the memory report the other
// tables print at upload.
//
// WHO READS IT, AND THAT IS WHY IT IS OFF BY DEFAULT TODAY
//
// The one consumer in this port's transport is the capture cascade:
// `G4NeutronRadCapture::ApplyYourself` hands its compound nucleus to
// `G4PhotonEvaporation::BreakUpChain`, which walks a nuclide's level scheme
// (`deexcitation/photon_evaporation.cuh`). That is a sub-process of
// `G4NeutronGeneralProcess`, which is NOT wired - see `physics/hadronic/neutron_general_xs.cuh`
// and `TransportEngine::Upload`'s refusal - so uploading the table in every run would cost
// every run something nothing reads.
//
// And the cost is not negligible: `read_all_level_data` opens **3110 files** over the AMIN/AMAX
// window, one per nuclide, and a B1 run's whole transport is 750 ms. So
// `TransportEngine::SetNuclearLevelData(true)` turns it on and the default is off, with the
// measured numbers printed when it does. When the neutron general process lands this becomes
// unconditional, which is what Geant4 does: `G4ExcitationHandler::SetParameters` calls
// `G4NuclearLevelData::UploadNuclearLevelData(Zmax+1)` at initialisation whether a neutron ever
// arrives or not.
//
// `tests/test_capture_device.cu` uses this uploader directly, so the path is exercised
// regardless of what the engine's default is.
#pragma once

#include <cstdio>
#include <string>

#include "data/level_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/deexcitation/corrections.cuh"

namespace g4gpu::host {

/// The device allocations behind a device-side `data::LevelTable`.
struct LevelTableOwner {
  data::LevelTable view{};
  void* managers = nullptr;
  void* levels = nullptr;
  void* transitions = nullptr;
  void* z_a_index = nullptr;
  std::size_t bytes = 0;
};

/// Reads PhotonEvaporation5.7 and uploads it. Returns an owner whose `view` is
/// device-dereferenceable, or one with a null view when the dataset cannot be found.
///
/// @param zmax passed straight to `read_all_level_data`, which loads `Z < min(zmax, ZMAX)` -
///        so `G4ExcitationHandler::SetParameters`' `Zmax + 1` convention is the caller's to
///        apply, as it is in Geant4. `data::kLevelZMax` loads the whole table.
/// @param storage where the host copy lives while it is uploaded. Kept by the caller because
///        a host-side comparison (which is what a device test is) needs it afterwards; pass a
///        local one to throw it away.
inline LevelTableOwner upload_level_data(data::LevelTableStorage& storage, int zmax,
                                          bool verbose = true) {
  LevelTableOwner own;
  const std::string& dir = g4photon_evaporation_dir();
  if (dir.empty()) {
    if (verbose) {
      std::printf("level data: G4LEVELGAMMADATA could not be resolved - the capture cascade "
                  "has no levels to walk\n");
    }
    return own;
  }
  // The two cached numbers of the G4LevelManager constructor, as callbacks so that
  // data/level_data.cuh stays free of the physics headers that depend on it.
  data::read_all_level_data(
      storage, dir, zmax,
      [](int Z, int A) { return deex::shell_correction(A, Z); },
      [](int Z, int A) {
        return deex::level_manager_level_density(Z, A);
      });
  const data::LevelTable host = storage.view();

  auto up = [&](const void* src, std::size_t n, void** out) {
    if (n == 0) { return; }
    if (cudaMalloc(out, n) != cudaSuccess) {
      std::printf("\nFATAL: could not allocate %zu bytes for the nuclear level table\n", n);
      std::exit(1);
    }
    if (cudaMemcpy(*out, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of the nuclear level table\n", n);
      std::exit(1);
    }
    own.bytes += n;
  };

  up(host.managers, sizeof(data::LevelManagerEntry) * host.n_managers, &own.managers);
  up(host.levels, sizeof(data::NuclearLevel) * host.n_levels, &own.levels);
  up(host.transitions, sizeof(data::LevelTransition) * host.n_transitions, &own.transitions);
  up(storage.z_a_index.data(), sizeof(int) * storage.z_a_index.size(), &own.z_a_index);

  own.view.managers = static_cast<const data::LevelManagerEntry*>(own.managers);
  own.view.levels = static_cast<const data::NuclearLevel*>(own.levels);
  own.view.transitions = static_cast<const data::LevelTransition*>(own.transitions);
  own.view.z_a_index = static_cast<const int*>(own.z_a_index);
  own.view.n_managers = host.n_managers;
  own.view.n_levels = host.n_levels;
  own.view.n_transitions = host.n_transitions;

  if (verbose) {
    std::printf("level data: %.2f MB - %d managers, %d levels (%zu B each), %d transitions "
                "(%zu B each), %zu index entries\n",
                double(own.bytes) / 1048576.0, host.n_managers, host.n_levels,
                sizeof(data::NuclearLevel), host.n_transitions,
                sizeof(data::LevelTransition), storage.z_a_index.size());
    if (!storage.notes.empty()) {
      std::printf("            %zu nuclide files were present and produced no manager; the "
                  "first is: %s\n", storage.notes.size(), storage.notes[0].c_str());
    }
  }
  return own;
}

inline void free_level_data(LevelTableOwner& own) {
  cudaFree(own.managers);
  cudaFree(own.levels);
  cudaFree(own.transitions);
  cudaFree(own.z_a_index);
  own = LevelTableOwner{};
}

}  // namespace g4gpu::host
