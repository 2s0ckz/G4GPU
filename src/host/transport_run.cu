// The transport engine's kernels and method bodies.
//
// These live in their own translation unit because they are __global__ templates. A
// __global__ template defined in a header emits a device stub into every translation unit
// that instantiates it, and nvcc then rejects the duplicates at compile time with
// "explicit specialization ... is not a specialization of a function template" - pointing at
// a generated stub file rather than at anything in this project. Keeping them here, with a
// single explicit instantiation at the bottom, is the standard answer.
#include <cstdio>
#include <cstdlib>

#include "host/transport_run.cuh"

namespace g4gpu::host {

// ---------------------------------------------------------------- kernels

/// Seeds the track buffers from primaries the host generated.
///
/// Replaces the kernel that sampled a Source record on the device. The reason is not
/// performance - that one was faster - but that it made the set of possible primaries closed:
/// a primary could only be whatever the gun's distributions could express. Generation happens
/// on the host now, once per event, in the user's own GeneratePrimaries. See
/// g4gpu::Primary and G4VUserPrimaryGeneratorAction.
///
/// The atomic is new and is the cost of the generality. The old kernel knew every primary was
/// the same species, so a track's slot was its event index and no counter was needed. Now an
/// event may start a gamma and the next an electron - a generator is free to do that - so each
/// species' buffer is appended to. One atomicAdd per event against a whole shower's transport
/// is not a cost worth designing around.
template <typename real_t>
__global__ void seed_from_primaries(TrackBuffer<real_t> gamma, TrackBuffer<real_t> electron,
                                    TrackBuffer<real_t> positron, TrackBuffer<real_t> proton,
                                    TrackBuffer<real_t> alpha,
                                    geom::Geometry<real_t> geometry,
                                    const Primary<real_t>* prim, int n, unsigned int seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const Primary<real_t> p = prim[i];

  TrackBuffer<real_t>* buf = &gamma;
  if (p.particle == ParticleType::kElectron) { buf = &electron; }
  else if (p.particle == ParticleType::kPositron) { buf = &positron; }
  else if (p.particle == ParticleType::kProton) { buf = &proton; }
  else if (p.particle == ParticleType::kAlpha) { buf = &alpha; }

  const int slot = atomicAdd(buf->count, 1);
  if (slot >= buf->capacity) {
    // Counted, never silent - the same discipline TrackBuffer::append keeps, and the run
    // reports it as an overflow rather than quietly transporting fewer primaries than asked.
    atomicAdd(buf->overflow, 1);
    return;
  }

  // The per-track RNG key still comes from the event index and the run's seed, so the shower
  // a primary produces is reproducible for a given seed even though the primary itself was
  // generated on the host from Geant4's engine.
  const unsigned int key = seed ^ static_cast<unsigned int>(i);
  buf->x[slot] = p.pos.x;   buf->y[slot] = p.pos.y;   buf->z[slot] = p.pos.z;
  buf->dx[slot] = p.dir.x;  buf->dy[slot] = p.dir.y;  buf->dz[slot] = p.dir.z;
  buf->ekin[slot] = p.ekin;
  buf->volume[slot] = geom::locate(geometry, p.pos);
  buf->event[slot] = i;
  buf->rng_key[slot] = key;
  buf->step[slot] = 0u;
  buf->msc_tlimit[slot] = real_t(0);
  buf->msc_tlimitmin[slot] = real_t(0);
}

template <typename real_t>
__global__ void run_step_gamma(Scene<real_t> scene, TrackBuffer<real_t> in,
                               TrackBuffer<real_t> g_out, TrackBuffer<real_t> e_out,
                               TrackBuffer<real_t> p_out, int n, int batch, double* score,
                               double* voxel_score,
                               vis::TrajectoryBuffer traj) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(i, p);
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // Which cell the step *starts* in, if this volume is scored per voxel. Taken before the
  // step, because the step moves p.pos - and it is the pre-step point that G4PSEnergyDeposit3D
  // keys its hits map by. The navigator ends a step at every cell boundary for such a volume,
  // so the whole deposit belongs to this one cell rather than being spread over several.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  Philox<real_t> rng(p.rng_key, p.step, 0u);
  BufferEmitter<real_t> em{g_out, e_out, p_out, p.pos, p.volume, p.event, p.rng_key, p.step, 0u};

  real_t edep = 0;
  const bool alive = step_gamma(scene, p, rng, em, edep, traj);
  ++p.step;
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  if (alive) { g_out.append(p); }
}

template <typename real_t, bool kIsPositron>
__global__ void run_step_lepton(Scene<real_t> scene, TrackBuffer<real_t> in,
                                TrackBuffer<real_t> g_out, TrackBuffer<real_t> e_out,
                                TrackBuffer<real_t> p_out, int n, int batch, double* score,
                                double* voxel_score,
                               vis::TrajectoryBuffer traj) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(i, p);
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // See the note in run_step_gamma: the cell the step *starts* in, which is what
  // G4PSEnergyDeposit3D keys its hits map by.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  Philox<real_t> rng(p.rng_key, p.step, 0x5A5Au);
  BufferEmitter<real_t> em{g_out, e_out, p_out, p.pos, p.volume, p.event, p.rng_key, p.step, 0u};

  real_t edep = 0;
  const bool alive = step_lepton(scene, p, kIsPositron, rng, em, edep, traj);
  ++p.step;
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  if (alive) {
    if (kIsPositron) { p_out.append(p); } else { e_out.append(p); }
  }
}

/// One step of one proton or alpha.
///
/// Templated on the species rather than reading it from the track, for the same reason the
/// lepton kernel is templated on kIsPositron: a buffer holds exactly one species, so the type
/// is a compile-time constant and no thread in a warp diverges on it.
template <typename real_t, ParticleType kType>
__global__ void run_step_hadron(Scene<real_t> scene, TrackBuffer<real_t> in,
                                TrackBuffer<real_t> g_out, TrackBuffer<real_t> e_out,
                                TrackBuffer<real_t> p_out, TrackBuffer<real_t> pr_out,
                                TrackBuffer<real_t> al_out, int n, int batch, double* score,
                                double* voxel_score, vis::TrajectoryBuffer traj) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  TrackState<real_t> p;
  in.load(i, p);
  const int slot = (p.volume >= 0) ? scene.geometry.volumes[p.volume].score_index : -1;

  // See the note in run_step_gamma: the cell the step *starts* in.
  int vcell = -1;
  if (slot >= 0 && p.volume >= 0 && voxel_score != nullptr) {
    const auto& vv = scene.geometry.volumes[p.volume];
    if (vv.score_per_voxel) {
      const auto grid = geom::voxel_grid_of(vv.solid);
      const auto q = geom::to_local(vv.xform, p.pos);
      int ijk[3];
      geom::voxel_cell_of(grid, q, ijk);
      vcell = grid.index(ijk[0], ijk[1], ijk[2]);
    }
  }

  // A third purpose value, so a hadron's stream is independent of the gamma and lepton streams
  // a track of the same key would have drawn.
  Philox<real_t> rng(p.rng_key, p.step, 0xB19Du);
  BufferEmitter<real_t> em{g_out, e_out, p_out, p.pos, p.volume, p.event, p.rng_key, p.step, 0u};

  real_t edep = 0;
  const bool alive = step_hadron(scene, p, kType, rng, em, edep, traj);
  ++p.step;
  if (edep != real_t(0) && slot >= 0) {
    atomicAdd(&score[static_cast<size_t>(slot) * batch + p.event], static_cast<double>(edep));
    if (vcell >= 0) { atomicAdd(&voxel_score[vcell], static_cast<double>(edep)); }
  }
  if (alive) {
    if (kType == ParticleType::kAlpha) { al_out.append(p); } else { pr_out.append(p); }
  }
}


// ---------------------------------------------------------------- method bodies

template <typename real_t>
void TransportEngine<real_t>::Upload(const g4::FlatScene& scene, int batch_size,
                                     int threads) {
    batch_ = batch_size;
    threads_ = threads;
    n_scorers_ = std::max<int>(1, static_cast<int>(G4SDManager::GetSDMpointer()->Scorers().size()));

    // The solid engine is mutually recursive, so the kernels need a real call stack; the 1 KB
    // default is not enough for one frame of the distance routine. Overflow surfaces as an
    // illegal memory access from an unrelated API call, with nothing pointing at the cause.
    G4GPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

    n_volumes_ = static_cast<int>(scene.volumes.size());
    h_mats_.assign(scene.materials.m, scene.materials.m + scene.materials.count);
    n_materials_ = scene.materials.count;

    // Solid pool, transforms and aux first: the volumes' boolean children index into them.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_solids_,
                                sizeof(geom::Solid<real_t>) * std::max<size_t>(1, scene.pool.solids.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_xforms_,
                                sizeof(geom::Transform<real_t>) * std::max<size_t>(1, scene.pool.xforms.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_pool_aux_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.aux.size())));
    if (!scene.pool.solids.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_solids_, scene.pool.solids.data(),
                                  sizeof(geom::Solid<real_t>) * scene.pool.solids.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.xforms.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_xforms_, scene.pool.xforms.data(),
                                  sizeof(geom::Transform<real_t>) * scene.pool.xforms.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.aux.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_pool_aux_, scene.pool.aux.data(),
                                  sizeof(real_t) * scene.pool.aux.size(),
                                  cudaMemcpyHostToDevice));
    }

    G4GPU_CUDA_CHECK(cudaMalloc(&d_vols_, sizeof(geom::Volume<real_t>) * n_volumes_));
    G4GPU_CUDA_CHECK(cudaMemcpy(d_vols_, scene.volumes.data(),
                                sizeof(geom::Volume<real_t>) * n_volumes_,
                                cudaMemcpyHostToDevice));

    G4GPU_CUDA_CHECK(cudaMalloc(&d_mats_, sizeof(data::Material<real_t>) * n_materials_));
    G4GPU_CUDA_CHECK(cudaMemcpy(d_mats_, h_mats_.data(),
                                sizeof(data::Material<real_t>) * n_materials_,
                                cudaMemcpyHostToDevice));

    em::build_range_table<real_t>(h_mats_.data(), h_rt_, nullptr, n_materials_);
    G4GPU_CUDA_CHECK(cudaMalloc(&d_rt_, sizeof(h_rt_)));

    geom::Geometry<real_t> g{};
    g.volumes = d_vols_;
    g.n_volumes = n_volumes_;
    g.world = scene.world;
    // Voxel cells: one pool for every voxel volume in the scene, indexed by each solid's
    // own offset. A 512^3 CT is 268 MB of shorts, so it is uploaded once here and never
    // touched again.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_voxels_,
                                sizeof(short) * std::max<size_t>(1, scene.pool.voxel_cells.size())));
    if (!scene.pool.voxel_cells.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_voxels_, scene.pool.voxel_cells.data(),
                                  sizeof(short) * scene.pool.voxel_cells.size(),
                                  cudaMemcpyHostToDevice));
    }

    // Mesh triangles and BVH nodes: one pool each for every tessellated solid in the scene,
    // with each solid's `a` naming its root node. A coarse CAD import is tens of thousands of
    // triangles at 72 bytes each, so this is uploaded once and never touched again.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_tri_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.tri.size())));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_bvh_,
                                sizeof(real_t) * std::max<size_t>(1, scene.pool.bvh.size())));
    if (!scene.pool.tri.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_tri_, scene.pool.tri.data(),
                                  sizeof(real_t) * scene.pool.tri.size(),
                                  cudaMemcpyHostToDevice));
    }
    if (!scene.pool.bvh.empty()) {
      G4GPU_CUDA_CHECK(cudaMemcpy(d_bvh_, scene.pool.bvh.data(),
                                  sizeof(real_t) * scene.pool.bvh.size(),
                                  cudaMemcpyHostToDevice));
    }

    g.store.solids = d_pool_solids_;
    g.store.xforms = d_pool_xforms_;
    g.store.aux = d_pool_aux_;
    g.store.tri = d_tri_;
    g.store.bvh = d_bvh_;
    g.voxels.material = d_voxels_;
    g.voxels.count = static_cast<int>(scene.pool.voxel_cells.size());
    geom_ = g;

    // Physics tables. The element list comes from the materials actually in the scene, so a
    // detector made of anything is covered without a hand-maintained list.
    std::vector<int> zs = DistinctZ(scene);
    auto* d_pe = upload_photoelectric_for<real_t>(default_phot_dir(), zs);
    auto brem = upload_brems_for<real_t>(default_sb_dir(), h_mats_.data(), n_materials_, h_rt_,
                                         zs);
    G4GPU_CUDA_CHECK(cudaMemcpy(d_rt_, &h_rt_, sizeof(h_rt_), cudaMemcpyHostToDevice));
    auto* d_ray = upload_rayleigh<real_t>(default_rayl_dir(), zs.data(),
                                          static_cast<int>(zs.size()));
    auto* d_msc = upload_msc<real_t>(h_mats_.data(), n_materials_);

    // The hadron range table. Built here rather than on demand because a run that will carry
    // a proton needs it before the first primary is seeded, and the engine cannot know what
    // species the generator will produce until it has produced one. Two megabytes in double
    // precision, which is not worth a conditional - the electron table beside it is larger.
    {
      auto* h = new em::HadronRangeTable<real_t>();
      std::vector<real_t> cuts(n_materials_);
      for (int i = 0; i < n_materials_; ++i) { cuts[i] = h_mats_[i].cut_electron; }
      static em::ShellTables<real_t> shell;
      em::build_shell_tables(shell);
      em::build_hadron_range_table<real_t>(h_mats_.data(), cuts.data(), *h, &shell,
                                           n_materials_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_hrt_, sizeof(*h)));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_hrt_, h, sizeof(*h), cudaMemcpyHostToDevice));
      delete h;
    }

    scene_ = Scene<real_t>{geom_,      d_mats_,  d_rt_,  d_pe,
                           brem.table, brem.sb,  d_ray,  d_msc,
                           d_hrt_,     static_cast<real_t>(scene.range_cut_mm),
                           -1,         processes_};

    for (int i = 0; i < 2; ++i) {
      gamma_[i].alloc(batch_ * 2);
      electron_[i].alloc(batch_ * 4);
      positron_[i].alloc(batch_ / 2 + 1024);
      // One slot per event plus slack. Nothing in the EM physics makes a hadron, so these
      // hold primaries and the survivors of one step of them - never a shower.
      proton_[i].alloc(batch_ + 1024);
      alpha_[i].alloc(batch_ + 1024);
    }
    G4GPU_CUDA_CHECK(cudaMalloc(&d_score_, sizeof(double) * n_scorers_ * batch_));

    // Per-cell scoring, allocated only when a volume actually asks for it. One double per
    // cell in the whole pool - the cell index is already global across voxel volumes, so no
    // per-volume offsets are needed here beyond the ones the store already carries.
    //
    // Summed over the run, not per event: the score array above is n_scorers by batch, and a
    // per-event copy of that shape for a 512-cubed grid would be a hundred million doubles a
    // batch to allocate and to copy back. A dose *distribution* is a run-level quantity; the
    // per-event statistics that the volume totals carry are what an uncertainty is computed
    // from, and those are unaffected.
    n_voxel_cells_ = 0;
    for (int i = 0; i < static_cast<int>(scene.volumes.size()); ++i) {
      if (scene.volumes[i].score_per_voxel) {
        n_voxel_cells_ = static_cast<int>(scene.pool.voxel_cells.size());
        break;
      }
    }
    if (n_voxel_cells_ > 0) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_voxel_score_, sizeof(double) * n_voxel_cells_));
      G4GPU_CUDA_CHECK(cudaMemset(d_voxel_score_, 0, sizeof(double) * n_voxel_cells_));
    }
  }


template <typename real_t>
RunStats TransportEngine<real_t>::BeamOn(int n_events, const Primary<real_t>* primaries,
                                         unsigned int seed,
                                         std::vector<double>& score_sum,
                                         std::vector<double>& score_sum_sq,
                                         vis::TrajectoryBuffer traj, EventSink* sink) {
    score_sum.assign(n_scorers_, 0.0);
    score_sum_sq.assign(n_scorers_, 0.0);
    // Zeroed per run, not per batch: this accumulates the whole run's deposit per cell.
    if (d_voxel_score_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_voxel_score_, 0, sizeof(double) * n_voxel_cells_));
    }
    RunStats st;
    if (primaries == nullptr || n_events <= 0) { return st; }

    // The three species the track buffers carry. A primary of any other species would land in
    // the gamma buffer by default and be transported as a gamma - a run that produced a
    // plausible dose for physics that never happened. Refuse it instead.
    //
    // Checked over every primary now rather than once for the source, because a generator is
    // free to fire a different species per event and the guard has to cover what it actually
    // produced. It is one comparison per event against a whole shower.
    for (int i = 0; i < n_events; ++i) {
      const ParticleType p = primaries[i].particle;
      if (p == ParticleType::kGamma || p == ParticleType::kElectron
          || p == ParticleType::kPositron) {
        continue;
      }
      if (p == ParticleType::kProton || p == ParticleType::kAlpha) {
        // Stepped by run_step_hadron, which needs the range table. A scene uploaded without
        // one - b1_gpu_sched builds a gamma-only Scene deliberately - would step the primary
        // once, find a null table, and drop it. That is a silent zero, so it is refused here.
        if (scene_.hadron_range == nullptr) {
          std::printf(
              "\nFATAL: a proton or alpha primary was given to a Scene with no hadron range\n"
              "  table (event %d). step_hadron cannot advance a track without one, and\n"
              "  dropping it would look like a run that simply deposited nothing.\n",
              i);
          std::exit(2);
        }
        continue;
      }
      std::printf(
          "\nFATAL: primary species %d is not transported (event %d).\n"
          "  This transport carries gamma, e-, e+, proton and alpha. Other hadrons and ions\n"
          "  are refused: see G4RunManager::CheckSpecies for what is missing for each, and\n"
          "  tests/test_hadron_range.cu for the measured size of the He3 gap. Refusing\n"
          "  rather than transporting it as something it is not.\n",
          static_cast<int>(p), i);
      std::exit(2);
    }

    // The primaries, uploaded a batch at a time. Sized once for the batch rather than for the
    // run: a ten-million-event run would otherwise want half a gigabyte resident for
    // something the device reads once.
    if (d_primaries_ == nullptr) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_primaries_, sizeof(Primary<real_t>) * batch_));
    }

    std::vector<double> ev_scores(static_cast<size_t>(n_scorers_), 0.0);

    std::vector<double> h_score(static_cast<size_t>(n_scorers_) * batch_);
    cudaEvent_t t0, t1;
    G4GPU_CUDA_CHECK(cudaEventCreate(&t0));
    G4GPU_CUDA_CHECK(cudaEventCreate(&t1));
    G4GPU_CUDA_CHECK(cudaEventRecord(t0));

    for (int base = 0; base < n_events; base += batch_) {
      const int n_batch = std::min(batch_, n_events - base);
      G4GPU_CUDA_CHECK(cudaMemset(d_score_, 0, sizeof(double) * n_scorers_ * batch_));

      int cur = 0, nxt = 1;
      gamma_[cur].reset();
      electron_[cur].reset();
      positron_[cur].reset();
      proton_[cur].reset();
      alpha_[cur].reset();
      G4GPU_CUDA_CHECK(cudaMemcpy(d_primaries_, primaries + base,
                                  sizeof(Primary<real_t>) * n_batch, cudaMemcpyHostToDevice));
      seed_from_primaries<real_t><<<(n_batch + threads_ - 1) / threads_, threads_>>>(
          gamma_[cur].view, electron_[cur].view, positron_[cur].view, proton_[cur].view,
          alpha_[cur].view, geom_, d_primaries_, n_batch, seed);
      G4GPU_CUDA_CHECK(cudaGetLastError());

      // Bounded so a non-terminating track shows up as an explicit abandonment rather than a
      // hang. B1 drains in about 110 iterations.
      constexpr int kMaxIterations = 1000;
      int leftover = 0;
      for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        const int ng = gamma_[cur].count();
        const int ne = electron_[cur].count();
        const int np = positron_[cur].count();
        const int nh = proton_[cur].count();
        const int na = alpha_[cur].count();
        if (ng + ne + np + nh + na == 0) {
          st.max_iterations = std::max(st.max_iterations, iteration);
          break;
        }
        if (iteration == kMaxIterations - 1) { leftover = ng + ne + np + nh + na; }
        st.track_steps += ng + ne + np + nh + na;
        st.peak_gamma = std::max(st.peak_gamma, ng);
        st.peak_electron = std::max(st.peak_electron, ne);
        st.peak_positron = std::max(st.peak_positron, np);
        st.peak_proton = std::max(st.peak_proton, nh);
        st.peak_alpha = std::max(st.peak_alpha, na);

        gamma_[nxt].reset();
        electron_[nxt].reset();
        positron_[nxt].reset();
        proton_[nxt].reset();
        alpha_[nxt].reset();

        if (ng > 0) {
          run_step_gamma<real_t><<<(ng + threads_ - 1) / threads_, threads_>>>(
              scene_, gamma_[cur].view, gamma_[nxt].view, electron_[nxt].view,
              positron_[nxt].view, ng, batch_, d_score_, d_voxel_score_, traj);
        }
        if (ne > 0) {
          run_step_lepton<real_t, false><<<(ne + threads_ - 1) / threads_, threads_>>>(
              scene_, electron_[cur].view, gamma_[nxt].view, electron_[nxt].view,
              positron_[nxt].view, ne, batch_, d_score_, d_voxel_score_, traj);
        }
        if (np > 0) {
          run_step_lepton<real_t, true><<<(np + threads_ - 1) / threads_, threads_>>>(
              scene_, positron_[cur].view, gamma_[nxt].view, electron_[nxt].view,
              positron_[nxt].view, np, batch_, d_score_, d_voxel_score_, traj);
        }
        if (nh > 0) {
          run_step_hadron<real_t, ParticleType::kProton>
              <<<(nh + threads_ - 1) / threads_, threads_>>>(
                  scene_, proton_[cur].view, gamma_[nxt].view, electron_[nxt].view,
                  positron_[nxt].view, proton_[nxt].view, alpha_[nxt].view, nh, batch_,
                  d_score_, d_voxel_score_, traj);
        }
        if (na > 0) {
          run_step_hadron<real_t, ParticleType::kAlpha>
              <<<(na + threads_ - 1) / threads_, threads_>>>(
                  scene_, alpha_[cur].view, gamma_[nxt].view, electron_[nxt].view,
                  positron_[nxt].view, proton_[nxt].view, alpha_[nxt].view, na, batch_,
                  d_score_, d_voxel_score_, traj);
        }
        G4GPU_CUDA_CHECK(cudaGetLastError());
        cur ^= 1;
        nxt ^= 1;
      }
      st.abandoned += leftover;

      G4GPU_CUDA_CHECK(cudaMemcpy(h_score.data(), d_score_,
                                  sizeof(double) * n_scorers_ * batch_,
                                  cudaMemcpyDeviceToHost));
      // Event-major, so that a sink sees one event's scores together. Each accumulator still
      // sums over events in ascending order, so the totals are bit-identical to the
      // scorer-major loop this replaced - which matters, because those totals are what the
      // 0.09-sigma agreement with Geant4 is measured on.
      for (int e = 0; e < n_batch; ++e) {
        for (int sc = 0; sc < n_scorers_; ++sc) {
          const double v = h_score[static_cast<size_t>(sc) * batch_ + e];
          score_sum[sc] += v;
          score_sum_sq[sc] += v * v;
          ev_scores[sc] = v;
        }
        if (sink != nullptr) { sink->Event(base + e, ev_scores.data(), n_scorers_); }
      }
    }

    // The per-cell deposit, brought back once at the end of the run.
    h_voxel_score_.clear();
    if (d_voxel_score_ != nullptr && n_voxel_cells_ > 0) {
      h_voxel_score_.resize(static_cast<std::size_t>(n_voxel_cells_));
      G4GPU_CUDA_CHECK(cudaMemcpy(h_voxel_score_.data(), d_voxel_score_,
                                  sizeof(double) * n_voxel_cells_, cudaMemcpyDeviceToHost));
    }

    G4GPU_CUDA_CHECK(cudaEventRecord(t1));
    G4GPU_CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    G4GPU_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    st.milliseconds = ms;
    st.overflow = gamma_[0].overflow() + electron_[0].overflow() + positron_[0].overflow()
                  + proton_[0].overflow() + alpha_[0].overflow()
                  + gamma_[1].overflow() + electron_[1].overflow() + positron_[1].overflow()
                  + proton_[1].overflow() + alpha_[1].overflow();
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return st;
  }



template <typename real_t>
void TransportEngine<real_t>::Free() {
    for (int i = 0; i < 2; ++i) {
      gamma_[i].free_all();
      electron_[i].free_all();
      positron_[i].free_all();
    }
    cudaFree(d_vols_);
    cudaFree(d_mats_);
    cudaFree(d_rt_);
    cudaFree(d_score_);
    cudaFree(d_primaries_);
    d_primaries_ = nullptr;
    cudaFree(d_voxel_score_);
    d_voxel_score_ = nullptr;
    n_voxel_cells_ = 0;
    cudaFree(d_pool_solids_);
    cudaFree(d_pool_xforms_);
    cudaFree(d_pool_aux_);
    cudaFree(d_voxels_);
    cudaFree(d_tri_);
    cudaFree(d_bvh_);
  }


template class TransportEngine<double>;

}  // namespace g4gpu::host
