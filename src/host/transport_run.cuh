// The transport engine: uploads a flattened scene and runs events on it.
//
// This is src/host/b1_gpu_sched.cu's loop, generalised - any number of volumes, materials and
// scorers, and primaries from a Source description rather than a hard-coded beam. The
// scheduling is unchanged and deliberately so: tracks live in structure-of-arrays buffers, one
// per species for warp coherence, advanced one step per kernel launch, with the output buffer
// compacting implicitly as survivors and secondaries are appended.
//
// Scoring note. A scorer slot is chosen from the volume the track occupied at the *start* of
// the step, which is where every deposit in the stepper is attributed: a step ends either at
// an interaction (deposit first, no crossing) or at a boundary (crossing last, after the
// continuous loss has been attributed). Using the post-step volume would move the continuous
// loss of a boundary-terminated step into the next volume.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "g4/G4Flatten.hh"
#include "host/pe_upload.cuh"
#include "physics/source.cuh"
#include "physics/stepper.cuh"
#include "render/trajectory.cuh"

namespace g4gpu::host {

#define G4GPU_CUDA_CHECK(call)                                                             \
  do {                                                                                     \
    const cudaError_t err__ = (call);                                                      \
    if (err__ != cudaSuccess) {                                                            \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__,         \
                  __LINE__);                                                               \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

template <typename real_t>
struct DeviceTracks {
  TrackBuffer<real_t> view{};

  void alloc(int capacity) {
    view.capacity = capacity;
    const size_t nr = sizeof(real_t) * capacity;
    G4GPU_CUDA_CHECK(cudaMalloc(&view.x, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.y, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.z, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.dx, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.dy, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.dz, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.ekin, nr));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.volume, sizeof(int) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.event, sizeof(int) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.rng_key, sizeof(unsigned int) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.msc_tlimit, sizeof(real_t) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.msc_tlimitmin, sizeof(real_t) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.step, sizeof(unsigned int) * capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.count, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMalloc(&view.overflow, sizeof(int)));
    reset();
    G4GPU_CUDA_CHECK(cudaMemset(view.overflow, 0, sizeof(int)));
  }
  void reset() { G4GPU_CUDA_CHECK(cudaMemset(view.count, 0, sizeof(int))); }
  int count() const {
    int c = 0;
    G4GPU_CUDA_CHECK(cudaMemcpy(&c, view.count, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
  }
  int overflow() const {
    int c = 0;
    G4GPU_CUDA_CHECK(cudaMemcpy(&c, view.overflow, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
  }
  void free_all() {
    cudaFree(view.x); cudaFree(view.y); cudaFree(view.z);
    cudaFree(view.dx); cudaFree(view.dy); cudaFree(view.dz);
    cudaFree(view.ekin); cudaFree(view.volume); cudaFree(view.event);
    cudaFree(view.rng_key); cudaFree(view.msc_tlimit); cudaFree(view.msc_tlimitmin);
    cudaFree(view.step); cudaFree(view.count); cudaFree(view.overflow);
    view = TrackBuffer<real_t>{};
  }
};

// ---------------------------------------------------------------- engine

struct RunStats {
  /// GPU time for the batch loop: primary upload, the stepping kernels, the score readback
  /// and the per-event sink callbacks. Measured with CUDA events, so it excludes everything
  /// the host did before the first kernel.
  double milliseconds = 0;
  /// Host wall clock for the whole event loop - the primaries generated one event at a time
  /// through the user's GeneratePrimaries, then the above. This is the number that compares
  /// against Geant4's, whose G4Timer brackets InitializeEventLoop to TerminateEventLoop and so
  /// also covers primary generation while excluding physics-table building. `milliseconds` is
  /// a strictly smaller number measuring a strictly smaller thing, and quoting it against
  /// Geant4's would be flattering rather than wrong.
  double event_loop_ms = 0;
  long long track_steps = 0;
  int max_iterations = 0;
  int peak_gamma = 0, peak_electron = 0, peak_positron = 0;
  int peak_proton = 0, peak_alpha = 0;
  long long abandoned = 0;
  int overflow = 0;
};

/// A per-event callback, so that a G4UserEventAction and a G4UserSteppingAction can be
/// called the way Geant4 calls them.
///
/// The scores are already per event on the device - d_score_ is n_scorers by batch, indexed
/// by the event's slot in the batch - so this costs a host virtual call per event and no
/// extra device work. At 2.7 million events per second that is a few per cent of the run,
/// which is why it is only installed when a user action asks for it.
struct EventSink {
  virtual ~EventSink() = default;
  /// @p scores has one entry per scorer, in device_index order, for event @p event_id.
  virtual void Event(int event_id, const double* scores, int n_scorers) = 0;
};

/// Owns the device-side scene and the track buffers, and runs events on them.
template <typename real_t>
class TransportEngine {
 public:
  /// Uploads @p scene. Materials were built with their production cuts already applied.
  void Upload(const g4::FlatScene& scene, int batch_size = 1048576, int threads = 128);

  /// Runs @p n_events from primaries the host generated, filling `score_sum` and
  /// `score_sum_sq` with one entry per scorer.
  ///
  /// @p primaries points at n_events records; the engine uploads them a batch at a time.
  /// They come from G4VUserPrimaryGeneratorAction::GeneratePrimaries, called once per event,
  /// so a primary can be anything a user's code can compute. The device used to sample a
  /// Source description instead, which was faster and closed the set of possible primaries to
  /// whatever distributions the gun offered - a Gaussian beam spot could not be written.
  ///
  /// @p seed keys the per-track RNG, so a given seed reproduces a run's showers.
  /// A non-empty @p traj captures trajectory segments for the first `traj.max_event` events.
  /// A non-null @p sink is called once per event with that event's scores.
  RunStats BeamOn(int n_events, const Primary<real_t>* primaries, unsigned int seed,
                  std::vector<double>& score_sum, std::vector<double>& score_sum_sq,
                  vis::TrajectoryBuffer traj = vis::TrajectoryBuffer{},
                  EventSink* sink = nullptr);

  void Free();

  const Scene<real_t>& scene() const { return scene_; }
  const geom::Geometry<real_t>& geometry() const { return geom_; }
  const std::vector<data::Material<real_t>>& materials() const { return h_mats_; }
  int batch() const { return batch_; }

  /// Energy deposit per voxel cell after the last BeamOn, in MeV, indexed by cell. Empty when
  /// no scorer asked for per-voxel scoring. Accumulated over the whole run rather than per
  /// event: a per-event copy of a 512-cubed grid is 130 million doubles a batch.
  const std::vector<double>& voxel_scores() const { return h_voxel_score_; }

  /// Which processes to run. Must be set before Upload; all on is the validated default.
  void SetProcesses(const ProcessFlags& f) { processes_ = f; }
  const ProcessFlags& GetProcesses() const { return processes_; }

 private:
  static std::vector<int> DistinctZ(const g4::FlatScene& scene) {
    std::vector<int> zs;
    for (int i = 0; i < scene.materials.count; ++i) {
      const auto& m = scene.materials.m[i];
      for (int e = 0; e < m.n_elements; ++e) {
        const int z = static_cast<int>(m.z[e] + 0.5);
        bool seen = false;
        for (int q : zs) {
          if (q == z) { seen = true; break; }
        }
        if (!seen) { zs.push_back(z); }
      }
    }
    std::sort(zs.begin(), zs.end());
    return zs;
  }

  int batch_ = 0, threads_ = 128, n_volumes_ = 0, n_materials_ = 0, n_scorers_ = 1;
  ProcessFlags processes_{};
  geom::Volume<real_t>* d_vols_ = nullptr;
  data::Material<real_t>* d_mats_ = nullptr;
  em::RangeTable<real_t>* d_rt_ = nullptr;
  geom::Solid<real_t>* d_pool_solids_ = nullptr;
  geom::Transform<real_t>* d_pool_xforms_ = nullptr;
  real_t* d_pool_aux_ = nullptr;
  short* d_voxels_ = nullptr;
  real_t* d_tri_ = nullptr;
  real_t* d_bvh_ = nullptr;
  double* d_score_ = nullptr;
  /// Energy deposit per voxel cell, summed over the whole run, or null when no scorer asks
  /// for it. One entry per cell in the scene's voxel pool, so a scene with no voxel volume
  /// allocates nothing.
  /// One batch of primaries, uploaded per batch. See BeamOn.
  Primary<real_t>* d_primaries_ = nullptr;
  double* d_voxel_score_ = nullptr;
  int n_voxel_cells_ = 0;
  em::RangeTable<real_t> h_rt_{};
  /// Built only when the scene can actually see a hadron - it is 256 bins x 2 species x
  /// every material, and a photon run has no use for it. Null in Scene when not built.
  em::HadronRangeTable<real_t>* d_hrt_ = nullptr;
  std::vector<data::Material<real_t>> h_mats_;
  std::vector<double> h_voxel_score_;
  geom::Geometry<real_t> geom_{};
  Scene<real_t> scene_{};
  DeviceTracks<real_t> gamma_[2], electron_[2], positron_[2];
  /// Sized far smaller than the lepton buffers: nothing in the EM physics creates a hadron,
  /// so these hold primaries and nothing else. One slot per event in the batch is exact.
  DeviceTracks<real_t> proton_[2], alpha_[2];
};


// The kernels this engine launches are __global__ templates. A __global__ template defined in
// a header gets a device stub emitted into every translation unit that instantiates it, and
// those stubs collide at link time ("is not a specialization of a function template"). So the
// kernels and these method bodies live in transport_run.cu, and the instantiation for double
// is declared here and defined there.
extern template class TransportEngine<double>;

}  // namespace g4gpu::host
