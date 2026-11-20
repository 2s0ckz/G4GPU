// The transport engine's host method bodies: the stack raise, `Upload`, `BeamOn` - the transport
// loop - and `Free`.
//
// THEY LIVED AT THE BOTTOM OF host/transport_run_impl.cuh UNTIL P32, and that header is read by
// every one of the engine's 27 translation units: the 26 that hold one kernel each (V65, V189)
// and transport_run.cu, which instantiates the class. So a change to the loop - which no kernel
// unit compiles a line of - made every unit stale, and an engine rebuild at one target is about
// 85 minutes. This file is included at the end of transport_run_impl.cuh unless
// G4GPU_ENGINE_KERNELS_ONLY is defined, and the kernel units define it; see the note there.
// Nothing else changed in the move: the bodies below are the ones that were there, byte for byte
// until P32's own changes to the loop (docs/RISK.md V283 onward).
//
// Not a header to include on its own: it is the second half of transport_run_impl.cuh and reads
// that file's kernels, macros and `extern template` block.
#pragma once
#include <chrono>

#include "host/transport_run_impl.cuh"
#include "host/flat_drain.cuh"

namespace g4gpu::host {

// P36's defaults (docs/RISK.md V307-V308): the flat pool's width, the straggler bound K in a
// cascade's own time steps, the hand-back list (a drain is at most the pending ring or one
// iteration's queue, both under it), and the deferred ring's full width when the pool is on.
inline constexpr int kFlatDefaultLanes = 16384;
inline constexpr int kFlatDefaultK = 64;
inline constexpr int kFlatStragCap = 1 << 18;
inline constexpr int kFlatDefaultWide = 8192;
// V310: the light-ion reaction's pool - V307: ~200 reactions a drain on the Fe56 beam, 21 time
// steps and 16 secondaries in flight each, at most 149 steps, so K = 160 hands none back.
inline constexpr int kFlatIonDefaultLanes = 2048;
inline constexpr int kFlatIonDefaultK = 160;

// P36 (docs/RISK.md V308-V311): the flat Binary cascade's pool and its run totals. Process
// statics rather than members, so that the engine class - and with it transport_run.cuh, which
// all twenty-seven units include - is not touched; a process runs one engine at a time.
inline FlatPool& flat_pool_() {
  static FlatPool p;
  return p;
}
inline FlatPool& flat_ion_pool_() {
  static FlatPool p;
  return p;
}
inline FlatStats& flat_ion_stats_() {
  static FlatStats s;
  return s;
}
inline FlatStats& flat_stats_() {
  static FlatStats s;
  return s;
}
/// `G4GPU_CASCADE_FLAT=0` selects the one-thread path for every Binary cascade.
/// `G4GPU_CASCADE_FLAT_WIDE=n`: the deferred ring's "full width" when the flat pool is on.
inline int flat_wide_cap_() {
  static const int v = [] {
    const char* e = std::getenv("G4GPU_CASCADE_FLAT_WIDE");
    return (e != nullptr) ? std::atoi(e) : kFlatDefaultWide;
  }();
  return v;
}
inline bool flat_requested_() {
  const char* v = std::getenv("G4GPU_CASCADE_FLAT");
  return !(v != nullptr && v[0] == '0');
}

// P34's deferred drain (docs/RISK.md V294; see `drain_interactions` in BeamOn): how many
// iterations a queued interaction may wait for a full-width drain, and the pending ring's
// entries (16,384 x 408 B = 6.7 MB, inside Upload's 512 MB margin).
inline constexpr int kDeferIterations = 256;
inline constexpr int kPendingCap = 16384;

// ---------------------------------------------------------------- the loop's own kernels
//
// The seeding kernel and the two counting sorts moved here from transport_run_impl.cuh with the
// method bodies (P32): only `BeamOn` launches them, they carry no physics and no hook, and they
// are instantiated implicitly in the one unit that instantiates the engine.


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
/// Where in the seed's random stream a batch starts: the run's own offset plus the batch's.
///
/// Wraps at 2^32, which is 4.3e9 primaries into a process's life and is the same wrap the key
/// arithmetic has always had. Folded into one function because a repair pass has to use the
/// same base as the attempt it repairs - re-transporting an event with a different key would
/// give it a different shower, and the dose would depend on whether a batch had to be
/// repaired.
inline unsigned int key_base_for(long long stream_pos, int batch_base) {
  return static_cast<unsigned int>(static_cast<unsigned long long>(stream_pos)
                                   + static_cast<unsigned long long>(batch_base));
}

template <typename real_t>
__global__ void seed_from_primaries(TrackBuffer<real_t> pool,
                                    geom::Geometry<real_t> geometry,
                                    const Primary<real_t>* prim, int n, unsigned int key_base,
                                    unsigned int seed, const int* only = nullptr,
                                    int first = 0) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) { return; }
  // `only` names the events to seed, for a repair pass that transports just the events
  // which lost a track. Without it every event in the batch is seeded, which is the normal
  // case and the one where `only` is null and this costs a predicated load.
  //
  // `first` (P32) is where in the window a refill starts: the refill seeds the window's events
  // `first .. first + n - 1`, each with ITS OWN window index - which is both the key offset and
  // the score column - so a primary seeded late is the track it would have been seeded early.
  const int i = (only != nullptr) ? only[t] : first + t;
  const Primary<real_t> p = prim[i];

  // No routing. A primary of any species goes into the one pool and says what it is; whether a
  // kernel exists to step it is checked once, here, rather than discovered by an if-chain that
  // silently fell through to gammas for anything it did not recognise.
  TrackBuffer<real_t>* buf = &pool;
  const int slot = atomicAdd(buf->count, 1);
  if (slot >= buf->capacity) {
    // Counted, never silent - the same discipline TrackBuffer::append keeps, and the run
    // reports it as an overflow rather than quietly transporting fewer primaries than asked.
    atomicAdd(buf->overflow, 1);
    return;
  }

  // The per-track RNG key comes from the run's seed and the primary's position in the random
  // stream, so the shower a primary produces is reproducible for a given seed even though the
  // primary itself was generated on the host from Geant4's engine.
  //
  // key_base is a position in the WHOLE stream, not in this batch and not in this run. Two
  // things fold into it and both were bugs before they were folded in:
  //
  //   the batch offset   With the batch-local index every batch replayed the same set of
  //                      streams - event 0 of the second batch drew exactly what event 0 of
  //                      the first had - so a 2M-event run in two batches was two correlated
  //                      halves rather than 2M independent showers. Unseen because the batch
  //                      was a fixed 1048576 and nothing varied it; automatic sizing changed
  //                      the batch and B1 moved from 0.019 to 0.450 sigma against Geant4.
  //
  //   the run offset     Without it, the second BeamOn of a process drew the same streams as
  //                      the first. B1 still differed between the two, because its gun draws
  //                      two host random numbers per event and the host engine carries on -
  //                      so the primaries differed while every shower repeated. A generator
  //                      that draws nothing repeated exactly. See G4RunManager::stream_pos_.
  //
  // Note `event[slot] = i` below: the SCORING index stays batch-local, because it indexes the
  // per-event score array. Only the key is global. Conflating the two is what made the first
  // of those two bugs invisible.
  const unsigned int key = seed ^ (key_base + static_cast<unsigned int>(i));
  buf->x[slot] = p.pos.x;   buf->y[slot] = p.pos.y;   buf->z[slot] = p.pos.z;
  buf->dx[slot] = p.dir.x;  buf->dy[slot] = p.dir.y;  buf->dz[slot] = p.dir.z;
  buf->ekin[slot] = p.ekin;
  buf->volume[slot] = geom::locate(geometry, p.pos);
  buf->event[slot] = i;
  buf->species[slot] = static_cast<int>(p.particle);
  buf->rng_key[slot] = key;
  buf->step[slot] = 0u;
  buf->msc_tlimit[slot] = real_t(0);
  buf->msc_tlimitmin[slot] = real_t(0);

  // The G4Track block for a primary. A primary has no parent and no creating process, which
  // is what Geant4 reports too - GetParentID() 0 and GetCreatorProcess() null - and its clock
  // starts at the vertex time the generator asked for rather than at zero, so a stepping
  // action reading GetGlobalTime() sees the gun's clock.
  //
  // The nuclide goes in with it (P30): a `kGenericIon` primary from `/gun/ion Z A` carries
  // `ion_za_of(Z, A)` on its record, and that one field is what makes it the same track a
  // secondary C12 is - see seed_track_slot. Zero for every other species.
  seed_track_slot(*buf, slot, p.pos, p.dir, p.ekin, buf->volume[slot], p.t0, p.ion_za);
}


///
/// This is what buys the single pool. Storage is agnostic - a track sits wherever there was
/// room and says what it is - but dispatch cannot be: a warp whose threads take different
/// physics paths serialises through all of them, so each specialised kernel has to launch over
/// a contiguous run of one species. The lists provide that run without the tracks themselves
/// being separated, which is the distinction the five buffers used to conflate.
///
/// One atomic per track into its species' cursor. Order within a list is not deterministic and
/// does not need to be: a track's random stream is keyed on its own key and step, not on where
/// it sits or what it sits beside.
///
/// A species with no kernel gets index -1 from species_index and is counted rather than
/// scattered. Nothing routes it somewhere plausible; the engine reports it.
///
/// TWO PASSES, NOT ONE, and one array instead of one per species. The previous version wrote
/// straight into `lists[sp]`, an array per species each sized at the whole pool - so naming a
/// sixteenth species cost `4 * pool` bytes a side whether any run ever produced one. A counting
/// sort pays four bytes a slot for any number of species: histogram here, prefix sum on the
/// host (it already reads the counts back to size the launches), scatter below.
///
/// P33 (docs/RISK.md V291): A SECONDARY KEY WITHIN A SPECIES - the material the track's volume
/// is made of, then its kinetic energy's binary exponent - so a warp's 32 tracks read
/// neighbouring rows of the same tables and are more likely to take the same branches. The
/// histogram is over (species, material, energy bin) buckets, `kSortKeys` a species, laid out
/// species-major, so each species' buckets are still one contiguous range of the list and the
/// launches are exactly what they were. The host reads the bucket counts back (it read the
/// species counts back before), turns them into the scatter's starting cursors and uploads
/// those; the scatter then bumps a bucket's cursor instead of a species'.
///
/// `G4GPU_SORT_KEYS` chooses the key, for the measurement: 1 (the default) material then
/// energy, 0 none - every track's sub-key is 0 and the sort is the species-only one it was -
/// 2 material alone, 3 energy alone. On the 2M-photon gate, best of three (V291): 0 513 ms,
/// 3 455 ms, 2 443 ms, 1 429-433 ms; half-octave energy bins under the material bought nothing
/// over the octave. The extra 20 KB read back and uploaded an iteration is inside the noise of
/// key 0 against the species-only sort it replaced (513-518 ms against 517-527).
///
/// Like the order within a bucket, which is still the atomics', none of this can move a number:
/// a track's random stream is keyed on its own key and step. The material is the volume's own
/// (a voxel grid's cells are not looked up: that would be a navigation call per track per
/// iteration for a sort key); a material index past `kSortMaterials - 1` shares the last.
constexpr int kSortMaterials = 8;
constexpr int kSortEnergyBins = 32;
constexpr int kSortKeys = kSortMaterials * kSortEnergyBins;

/// The bucket within a species: 0 when the keyed sort is off. The energy bin is
/// `ilogb(ekin) + 10` clamped to [0, 31] - a binary decade from 1 keV (2^-10 MeV) to 2 TeV,
/// one integer instruction rather than a logarithm.
template <typename real_t>
__device__ __forceinline__ int sort_subkey(const TrackBuffer<real_t>& pool,
                                           const geom::Volume<real_t>* volumes, int i,
                                           int keyed) {
  if (keyed == 0) { return 0; }
  const int v = pool.volume[i];
  int m = (v >= 0) ? volumes[v].material : kSortMaterials - 1;
  m = (m < 0 || m >= kSortMaterials) ? kSortMaterials - 1 : m;
  if (keyed == 2) { return m * kSortEnergyBins; }
  const real_t e = pool.ekin[i];
  int eb = (e > real_t(0)) ? ilogb(e) + 10 : 0;
  eb = eb < 0 ? 0 : (eb >= kSortEnergyBins ? kSortEnergyBins - 1 : eb);
  if (keyed == 3) { return eb; }
  return m * kSortEnergyBins + eb;
}

template <typename real_t>
__global__ void count_species(TrackBuffer<real_t> pool, int n,
                              const geom::Volume<real_t>* volumes, int keyed, int* counts,
                              int* unknown) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int sp = species_index(static_cast<ParticleType>(pool.species[i]));
  if (sp < 0) {
    // A tripwire, not an accounting line. BufferEmitter::push refuses a species with no kernel
    // before it can be appended, and BeamOn refuses one as a primary, so a track reaching here
    // got past both - which is a hole in the guards rather than a particle to be counted.
    atomicAdd(unknown, 1);
    return;
  }
  atomicAdd(&counts[sp * kSortKeys + sort_subkey(pool, volumes, i, keyed)], 1);
}

/// Start of each species' contiguous range in the one index list.
struct SpeciesOffsets {
  int base[kNumTrackSpecies];
};

/// Writes each track's pool slot into its own bucket's range.
///
/// `cursors` is the same allocation the histogram used, overwritten by the host with each
/// bucket's starting position - a bump counter now rather than a count. Reusing it is not a
/// saving worth making on its own; what it avoids is a second buffer of the same size whose
/// only distinguishing feature would be which of the two passes wrote it.
template <typename real_t>
__global__ void scatter_species(TrackBuffer<real_t> pool, int n,
                                const geom::Volume<real_t>* volumes, int keyed, int* list,
                                int* cursors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int sp = species_index(static_cast<ParticleType>(pool.species[i]));
  if (sp < 0) { return; }  // already counted by count_species
  list[atomicAdd(&cursors[sp * kSortKeys + sort_subkey(pool, volumes, i, keyed)], 1)] = i;
}

/// P37 (docs/RISK.md V304): the host's prefix sum over the bucket counts, on the card.
///
/// The same exclusive sum in the same species-major order - each bucket's count becomes its
/// starting cursor, so a species' buckets are one contiguous range from `base[sp]` - computed
/// by one block of `kSortScanThreads`, each thread summing `kSortScanPer` consecutive buckets
/// and the block scanning the thread sums. Integer adds, so the cursors are the host's to the
/// bit whatever the order they are summed in. `summary` receives `base[sp]` and then the
/// species' track counts, the 40 integers the host needs to size the launches - which is all
/// that crosses the bus now, where the 5,120 counts went down and their 5,120 cursors came back.
constexpr int kSortScanThreads = 1024;
constexpr int kSortScanPer = (kNumTrackSpecies * kSortKeys + kSortScanThreads - 1) / kSortScanThreads;
__global__ void __launch_bounds__(kSortScanThreads) scan_species_buckets(int* counts,
                                                                         int* summary,
                                                                         int* host_out) {
  constexpr int kBuckets = kNumTrackSpecies * kSortKeys;
  __shared__ int part[kSortScanThreads];
  const int t = threadIdx.x;
  const int b0 = t * kSortScanPer;
  int own = 0;
  for (int k = 0; k < kSortScanPer; ++k) {
    const int b = b0 + k;
    if (b < kBuckets) { own += counts[b]; }
  }
  part[t] = own;
  __syncthreads();
  for (int d = 1; d < kSortScanThreads; d <<= 1) {
    const int v = (t >= d) ? part[t - d] : 0;
    __syncthreads();
    part[t] += v;
    __syncthreads();
  }
  int at = part[t] - own;  // exclusive
  for (int k = 0; k < kSortScanPer; ++k) {
    const int b = b0 + k;
    if (b < kBuckets) {
      const int c = counts[b];
      counts[b] = at;
      if (b % kSortKeys == 0) { summary[b / kSortKeys] = at; }
      at += c;
    }
  }
  __syncthreads();
  if (t < kNumTrackSpecies) {
    const int next = (t + 1 < kNumTrackSpecies) ? summary[t + 1] : part[kSortScanThreads - 1];
    summary[kNumTrackSpecies + t] = next - summary[t];
  }
  __syncthreads();
  if (host_out != nullptr && t < 2 * kNumTrackSpecies) { host_out[t] = summary[t]; }
}

/// P37 (V304): one device integer into mapped pinned host memory, the loop's read-back of an
/// append cursor. On this card under WDDM a `cudaMemcpy` of four bytes after a launch costs 62 us
/// and the launch plus a synchronisation 25 us (V304's probe); a kernel writing the host's
/// memory and a stream synchronisation is the second.
__global__ void copy_int_to_host(const int* src, int* host_out) {
  if (threadIdx.x == 0) { *host_out = *src; }
}

// ---------------------------------------------------------------- P15: binning the queue
//
// `count_species` and `scatter_species` again, over the interaction queue instead of the track
// pool, and for the same two reasons: a kernel can only be launched over a contiguous range,
// and a warp whose threads take different physics paths serialises through all of them. The
// second reason is stronger here than it is for the steppers - "different physics paths" is
// Bertini against FTFP rather than two multiple-scattering models - and so is the first, because
// the kernels are in different translation units and a thread cannot choose between them at all.
//
// Both carry no physics and are templated on `real_t` alone, so they stay in the engine's own
// object like the three utility kernels above.
template <typename real_t>
__global__ void count_interactions(const had::PendingInteraction<real_t>* items, int n,
                                   int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  atomicAdd(&counts[static_cast<int>(items[i].bucket)], 1);
}

/// Start of each bucket's contiguous range in the one index list.
struct InteractionOffsets {
  int base[static_cast<int>(had::InteractionBucket::kNumInteractionBuckets)];
};

template <typename real_t>
__global__ void scatter_interactions(const had::PendingInteraction<real_t>* items, int n,
                                     int* list, InteractionOffsets off, int* cursors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const int b = static_cast<int>(items[i].bucket);
  list[off.base[b] + atomicAdd(&cursors[b], 1)] = i;
}


// ---------------------------------------------------------------- method bodies

/// Raises the per-thread device stack to what the interaction kernels need, once, the first
/// time an iteration QUEUES an interaction (since P19: the first time it had a hadron in it
/// before that - which a photon run's photo-nuclear neutron now satisfies with nothing queued).
///
/// WHY IT IS DEFERRED AT ALL is in `Upload`'s own comment and in docs/RISK.md V190: 86,016
/// bytes a thread reserves five of an 8 GB card's gigabytes and costs B1's 6 MeV gamma gate
/// 28% of its throughput, on a run that can never reach a hadronic model.
///
/// WHY IT IS FATAL WHEN IT FAILS. The reservation is made after the pools are allocated, so it
/// can fail for want of memory where the same call at `Upload` would have succeeded. There is
/// no safe fall-back: a kernel whose frame is larger than the limit does not warn, it takes an
/// illegal memory access, and the run after it reports a plausible number. Lowering the slot
/// count or the batch is what a caller does about it, and the message says so.
template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::RaiseStackForInteractions() {
  if (stack_raised_) { return; }
  stack_raised_ = true;
  // THE NUMBER IS READ OFF THE FIVE KERNELS, and until P15's second pass it was written down -
  // 86,016, "81,584 rounded up to a 4 kB boundary with one page of margin". A constant like that
  // is right for exactly one build: P9e's `G4BinaryLightIonReaction::Interact` puts a whole
  // Binary cascade behind the light-ion arm, and its own probe measured 63,088 bytes of frame
  // for that entry point alone (V183 on its branch), against the 31,952 `run_interaction<
  // kLightIon>` had when the constant was chosen. So the reservation is the largest
  // `localSizeBytes` of the five, rounded up the same way.
  //
  // WHY `localSizeBytes` IS THE RIGHT QUANTITY, measured rather than assumed (docs/RISK.md
  // V196). For a call graph ptxas can size, it overlays every non-inlined callee's frame INTO
  // THE ENTRY'S: `-Xptxas -v` on `transport_run_int_binary` reports `propagate`,
  // `do_time_step`, `apply_collision`, `bic_deexcite_fragment` and P6's `deexcite` at 0 bytes
  // each and the kernel at 81,584, and `cudaFuncGetAttributes` returns the entry's number. So
  // the entry's frame IS the tree's requirement, and V190's "maximum over the call tree" was
  // right.
  //
  // AND WHAT A SHORTFALL WOULD COST, which is less than this function's first version claimed.
  // For a kernel ptxas can size, this driver (610.62, CUDA 11.6) RAISES THE LIMIT ITSELF at the
  // launch: a probe needing 40,000 B, launched under a 1,024 B limit, ran to the right answer and
  // left the limit at 40,000. So a reservation that came up short would not fault - it would
  // spend the memory at an interaction launch nobody chose, after the pools were sized against
  // the free memory printed below. That is still a defect, because every 4 kB is 270 MB of an
  // 8 GB card, and the report checks for it: `stack_reserved_` against the limit after the run.
  struct KernelFrame {
    const char* name;
    cudaFuncAttributes attr;
  };
  KernelFrame k[5] = {{"run_interaction<kFtfp>", {}},     {"run_interaction<kBertini>", {}},
                      {"run_interaction<kBinary>", {}},   {"run_interaction<kLightIon>", {}},
                      {"run_interaction<kAtRest>", {}}};
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[0].attr, run_interaction<real_t, had::InteractionBucket::kFtfp, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[1].attr, run_interaction<real_t, had::InteractionBucket::kBertini, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[2].attr, run_interaction<real_t, had::InteractionBucket::kBinary, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[3].attr, run_interaction<real_t, had::InteractionBucket::kLightIon, StepHook>));
  G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
      &k[4].attr, run_interaction<real_t, had::InteractionBucket::kAtRest, StepHook>));
  std::size_t need = 0;
  for (const KernelFrame& kf : k) {
    if (kf.attr.localSizeBytes > need) {
      need = kf.attr.localSizeBytes;
      stack_kernel_ = kf.name;
    }
  }
  // V190's rule: up to a 4 kB boundary, and one page of margin on top. Never below the
  // stepping kernels' floor, because the stepping kernels run under the same limit and theirs
  // is the one the driver cannot enforce for them.
  std::size_t want = ((need + 4095u) / 4096u) * 4096u + 4096u;
  want = std::max(want, stepping_stack_bytes());
  std::size_t have = 0;
  cudaDeviceGetLimit(&have, cudaLimitStackSize);
  stack_reserved_ = std::max(want, have);
  if (have >= want) { return; }
  // THE CARD MUST HOLD IT, or the raise is refused by name (docs/RISK.md V258). The driver
  // reserves the limit for every thread the card can hold resident, so raising it from `have`
  // to `want` asks `(want - have)` bytes of each of them - V190's 270 MB per 4 kB on the 3070's
  // 70,656 threads, V239's 11.9 GB on the Titan V's 163,840 - and WDDM does not fail a request
  // the card cannot hold: it pages it through system memory. So it is read here, before the
  // call, against what is free now that the pools and the queues are allocated.
  {
    const host::SelectedDevice& dev = host::select_device();
    const long long threads = static_cast<long long>(dev.prop.multiProcessorCount) *
                              static_cast<long long>(dev.prop.maxThreadsPerMultiProcessor);
    size_t f0 = 0, t0 = 0;
    cudaMemGetInfo(&f0, &t0);
    host::PagingRequest r;
    r.what = "the device stack raise for the interaction kernels";
    r.asked = (want - have) * static_cast<std::size_t>(threads);
    r.free_bytes = f0;
    r.total_bytes = t0;
    r.card = dev.prop.name;
    r.resident_threads = threads;
    r.frame = need;
    r.per_thread = want;
    host::refuse_if_paging(r);
  }
  const cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, want);
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  if (e != cudaSuccess) {
    std::printf("\nFATAL: the first hadron of this run needs %zu bytes of device stack a\n"
                "       thread - %s's frame is %zu - and the driver refused: %s.\n"
                "       %.2f GB of %.2f GB is free. Lower the batch, lower\n"
                "       SetInteractionSlots (now %d, %.1f MB), or run without hadrons.\n"
                "       docs/RISK.md V190 and V196.\n",
                want, stack_kernel_, need, cudaGetErrorString(e), double(f) / 1073741824.0,
                double(t) / 1073741824.0, n_interaction_slots_,
                double(interaction_bytes_) / 1048576.0);
    std::exit(2);
  }
  std::printf("device stack raised to %zu B a thread for the interaction kernels - %s's frame is "
              "%zu B - %.2f GB of %.2f GB left (docs/RISK.md V190, V196)\n",
              want, stack_kernel_, need, double(f) / 1073741824.0, double(t) / 1073741824.0);
}

template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::Upload(const g4::FlatScene& scene, int batch_size,
                                     int threads) {
    // The card first, before any allocation binds the context: G4GPU_DEVICE chooses it once for
    // the process (host/device_select.cuh), device 0 when unset.
    select_device();
    batch_ = batch_size;
    // P32: `G4GPU_BATCH=n` sets the batch for any program, as `G4GPU_LIVE_PER_EVENT` sets the
    // pool: examples/B1 sizes it automatically and has no command that reaches the setter, and a
    // run of several batches is what the batch refill is for (docs/RISK.md V284). The generator
    // is keyed per primary, so the batch moves time, not doses - the pool gate's three sizes
    // already hold that, and V284 holds it again for a batch smaller than the run.
    if (const char* env = std::getenv("G4GPU_BATCH")) {
      const int v = std::atoi(env);
      if (v > 0) { batch_ = v; }
    }
    threads_ = threads;
    n_scorers_ = std::max<int>(1, static_cast<int>(G4SDManager::GetSDMpointer()->Scorers().size()));

    // The solid engine is mutually recursive, so the kernels need a real call stack; the 1 KB
    // default is not enough for one frame of the distance routine. Overflow surfaces as an
    // illegal memory access from an unrelated API call, with nothing pointing at the cause.
    //
    // **86,016 SINCE P15, AND EVERY DIGIT OF IT IS A MEASUREMENT.** 16,384 was the stepping
    // kernels' ceiling and they are still far under it (`run_step_hadron<kProton>` is 3,936 B).
    // The interaction kernels are not. `-Xptxas -v` on the engine's own five units:
    //
    //     run_interaction<kFtfp>       23,008 B      run_interaction<kAtRest>     35,360 B
    //     run_interaction<kLightIon>   31,968 B      run_interaction<kBinary>   **81,600 B**
    //     run_interaction<kBertini>    33,680 B
    //
    // A kernel's frame is the maximum over its call tree, and the Binary cascade's tree -
    // `bic::apply_yourself` through `Propagate`, `ApplyCollision` and `DeExcite` into P6's
    // PreCompound and P3's evaporation - is 2.4 times the next largest. A limit below the frame
    // is not a warning: it is an illegal memory access from an unrelated API call, which is
    // exactly the failure this line was written for in the first place.
    //
    // **AND IT IS EXPENSIVE, DEVICE-WIDE, AND NOT BOUNDED BY THE LAUNCH.** CUDA reserves the
    // stack for the maximum threads the DEVICE can hold resident, not for the threads a kernel
    // is launched with, so `SetInteractionSlots` does not bound it. Measured on this RTX 3070
    // (46 SMs, 1536 threads an SM, 8.00 GB) with `cudaMemGetInfo` after the limit is set:
    //
    //      16,384 B/thread   5.938 GB free      86,016 B/thread   0.952 GB free
    //      81,920 B/thread   1.221 GB free      98,304 B/thread   0.099 GB free
    //
    // So the Binary cascade's frame costs this port **five of the card's eight gigabytes**, and
    // 86,016 is 81,600 rounded up to a 4 kB boundary with one page of margin - not a round
    // number chosen for comfort, because every 4 kB past it is another 270 MB. What is left is
    // what `SetInteractionSlots` and the batch have to fit in, which is why the pool's default
    // dropped to 128 slots and why `Upload` prints the free memory after both.
    // `G4GPU_STACK_BYTES` overrides it, for the same reason `G4GPU_LIVE_PER_EVENT` overrides
    // the pool: the only way to answer "what did this cost" is to run the same binary both
    // ways, and a program that has no such option cannot be asked. It is a RESOURCE knob and
    // not a physics one - but setting it below a kernel's frame is an illegal memory access in
    // that kernel, not a warning, so anything under 86,016 is only for a run that cannot make a
    // hadron.
    // **RAISED LAZILY, ON THE FIRST ITERATION THAT HAS A HADRON IN IT.** `Upload` sets the
    // stepping kernels' 16,384 and `RaiseStackForInteractions` sets the interaction kernels'
    // 86,016 the first time one could run, because the difference is 28% of a photon run's
    // throughput and a photon run can never reach a hadronic model.
    //
    // MEASURED, on B1's own 2,000,000-event 6 MeV gamma gate, same binary, one environment
    // variable apart:
    //
    //     16,384 B/thread   1,160 ms of event loop   1.72e6 events/s   5.94 GB free
    //     86,016 B/thread   1,616 ms                 1.24e6 events/s   1.17 GB free
    //
    // and the dose is 425.86 pGy either way, to every digit it prints. So this is not physics
    // and it is not memory pressure - halving the track pool moved it by 1.4% - it is what a
    // device-wide stack reservation does to occupancy and local-memory addressing.
    //
    // `G4GPU_STACK_BYTES` overrides both, for the same reason `G4GPU_LIVE_PER_EVENT` overrides
    // the pool: the only way to answer "what did this cost" is to run the same binary both ways.
    //
    // **TWO THINGS ABOVE ARE CORRECTED BY V196, AND THE TABLE IS HISTORY.** The frames are the
    // ones this comment was written against; `RaiseStackForInteractions` now READS the number
    // off the five kernels, so it cannot go stale when P9e's `Interact` grows the light-ion
    // arm. And "a limit below the frame is an illegal memory access" is true of THIS limit - the
    // stepping kernels', whose stack ptxas cannot size because the solid engine recurses - and
    // not of the interaction kernels', which ptxas sizes exactly and which this driver simply
    // raises the limit for at launch if nothing else has. Both measured, with two probes of
    // thirty lines each, in docs/RISK.md V196.
    G4GPU_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stepping_stack_bytes()));

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

    // The allocation only, here: the e+- dE/dx table is the sum over G4eIonisation AND
    // G4eBremsstrahlung, so it cannot be built until the Seltzer-Berger data is loaded, which
    // happens below in `upload_brems_for`. This used to build it with a crude radiative-yield
    // scaling and then throw that away - a table nothing could read, built twice.
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

    // PER-CLASS LAYERS, and only when some class asked for one.
    //
    // The class array is another short per cell - 268 MB for a 512^3 phantom, the same as the
    // materials - so a batch run whose classes all sit on one layer must not be made to carry
    // it. `any_class_layers` is the scene saying whether the navigator will ever look.
    // Either reason: a class can be absent without any class naming a layer, and the
    // navigator still has to read the per-cell class array to know which cells are gone.
    if ((scene.pool.any_class_layers || scene.pool.any_class_absent)
        && scene.pool.voxel_class_cells.size() == scene.pool.voxel_cells.size()
        && !scene.pool.voxel_class_layer.empty()) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_voxel_class_,
                                  sizeof(short) * scene.pool.voxel_class_cells.size()));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_voxel_class_, scene.pool.voxel_class_cells.data(),
                                  sizeof(short) * scene.pool.voxel_class_cells.size(),
                                  cudaMemcpyHostToDevice));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_class_layer_,
                                  sizeof(int) * scene.pool.voxel_class_layer.size()));
      G4GPU_CUDA_CHECK(cudaMemcpy(d_class_layer_, scene.pool.voxel_class_layer.data(),
                                  sizeof(int) * scene.pool.voxel_class_layer.size(),
                                  cudaMemcpyHostToDevice));
      g.voxels.cls = d_voxel_class_;
      g.voxels.class_layer = d_class_layer_;
      // One byte per CLASS, not per cell, so this is bounded by the class cap however large
      // the phantom is - and it is uploaded only when some class is actually absent.
      if (scene.pool.any_class_absent
          && scene.pool.voxel_class_absent.size() == scene.pool.voxel_class_layer.size()) {
        G4GPU_CUDA_CHECK(cudaMalloc(&d_class_absent_,
                                    scene.pool.voxel_class_absent.size()));
        G4GPU_CUDA_CHECK(cudaMemcpy(d_class_absent_, scene.pool.voxel_class_absent.data(),
                                    scene.pool.voxel_class_absent.size(),
                                    cudaMemcpyHostToDevice));
        g.voxels.class_absent = d_class_absent_;
      }
    }
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
    // The other half of the lepton's multiple scattering. Urban's table above stops at
    // `G4EmParameters::MscEnergyLimit()` because that is where Geant4's Urban model stops;
    // this one starts there and runs to 100 TeV. Unconditional: a run that makes any lepton
    // can make one above 100 MeV, and `step_lepton` reads it without a null check for the
    // reason Scene's field says.
    d_wv_ = upload_wv_lepton<real_t>(h_mats_.data(), n_materials_);

    // hadElastic's device tables, beside the hadron range table and for the same reason: a run
    // that will carry a charged hadron needs them before the first primary is seeded. The
    // element list is the scene's, so the per-(pion, Z) G4ElasticData is built for the elements
    // the detector is actually made of rather than for all 92. See host/hadronic_upload.cuh.
    elastic_tables_ = upload_elastic_tables<real_t>(zs);

    // The neutron's cross sections. Both configurations, because the stage is a run-time switch
    // and the same binary has to produce both columns of the comparison table: the two
    // per-process G4PARTICLEXS data sets for `kStage1` and G4NeutronGeneralProcess's five
    // combined tables for `kFinal`. The combined ones are per SCENE, not per element, because
    // `BuildPhysicsTable` sums MACROSCOPIC cross sections and so needs the atom densities.
    neutron_tables_ = upload_neutron_tables<real_t>(h_mats_.data(), n_materials_);
    d_neutron_xs_ = neutron_tables_.d_general;

    // P15: the five G4ParticleInelasticXS data sets and G4BGGPionInelasticXS. Beside the
    // elastic tables and for the same reason - a run that will carry a charged hadron needs
    // them before the first primary is seeded, and the engine cannot know what the generator
    // will produce until it has produced one.
    inelastic_tables_ = upload_inelastic_tables<real_t>();

    // P19: G4EmExtraPhysics' tables - the photon's G4GammaNuclearXS, the muon's Kokoulin
    // vectors and G4MuonVDNuclearModel's sampling table - and the per-material thresholds below
    // which the photo- and electro-nuclear cross sections are exactly zero. Unconditional for
    // the reason the inelastic ones are: a photon beam's shower can reach the giant resonance,
    // and `Upload` cannot know whether it will. See host/hadronic_upload.cuh.
    //
    // P21: and G4GammaGeneralProcess's table 9, which is built from the photoelectric cross
    // section as well as the store's - so the photoelectric data are read once more here, onto
    // the HOST, for the scene's elements. A missing file is fatal exactly as it is above.
    {
      data::PhotoElectricTable<real_t> pe_host{};
      std::vector<real_t> pe_e, pe_v;
      if (!data::load_photoelectric<real_t>(default_phot_dir(), zs.data(),
                                            static_cast<int>(zs.size()), pe_host, pe_e, pe_v)) {
        fatal_missing_data("photoelectric (G4EMLOW epics2017/phot), for table 9",
                           default_phot_dir());
      }
      pe_host.table_e = pe_e.data();
      pe_host.table_v = pe_v.data();
      pe_host.table_n = static_cast<int>(pe_e.size());
      emextra_tables_ =
          upload_emextra_tables<real_t>(h_mats_.data(), n_materials_, true, &pe_host);
    }

    // P3's nuclear level data, which the capture sub-process walks. Unconditional by default
    // since P8d - `SetNuclearLevelData` has the reason, and it is Geant4's own answer
    // (`G4ExcitationHandler::SetParameters` at initialisation, whether a neutron arrives or
    // not).
    if (load_level_data_) {
      // `Zmax + 1`, which is the convention G4ExcitationHandler::SetParameters applies -
      // `UploadNuclearLevelData(Zmax+1)` - and `read_all_level_data`'s strict `Z < mZ` is why
      // the element with the largest Z in the geometry is loaded only because of the +1.
      int zmax = 20;
      for (int z : zs) {
        if (z > zmax) { zmax = z; }
      }
      level_tables_ = upload_level_data(level_storage_, zmax + 1);
    }

    // The hadron range table. Built here rather than on demand because a run that will carry
    // a proton needs it before the first primary is seeded, and the engine cannot know what
    // species the generator will produce until it has produced one. A megabyte in double
    // precision - ten species against the electron table's two - which is not worth a
    // conditional beside the geometry and the photon data.
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
                           d_wv_,      d_hrt_,
                           static_cast<real_t>(scene.range_cut_mm),
                           -1,         processes_};

    // ---- how big a batch, and who decided.
    //
    // A batch needs pool_ = batch * live_per_event slots on each side of the ping-pong, and
    // track_arena_half_bytes() in core/track_buffer.cuh turns that into bytes by dry-running
    // the same allocator the carve then walks. One function, so the number that picks the
    // batch and the number that is consumed cannot disagree.
    // ---- P36: the flat Binary cascade's pool (docs/RISK.md V308, V311), BEFORE the batch, so
    // the track buffers are sized from what it leaves. `G4GPU_CASCADE_FLAT_LANES=n` sets its
    // width; `G4GPU_CASCADE_FLAT_K` the straggler bound. A hook project's engine keeps the
    // thread path (its interaction units are its own), and so does a run that asks for it.
    if constexpr (std::is_same<StepHook, StepTap<double>>::value &&
                  std::is_same<real_t, double>::value) {
      if (flat_requested_() && !flat_pool_().ok()) {
        int lanes = kFlatDefaultLanes;
        int k = kFlatDefaultK;
        if (const char* v = std::getenv("G4GPU_CASCADE_FLAT_LANES")) { lanes = std::atoi(v); }
        if (const char* v = std::getenv("G4GPU_CASCADE_FLAT_K")) { k = std::atoi(v); }
        if (lanes > 0) {
          const std::size_t want = flat_binary_lane_bytes() * static_cast<std::size_t>(lanes);
          size_t ff = 0, ft = 0;
          cudaMemGetInfo(&ff, &ft);
          host::PagingRequest r;
          r.what = "the flat Binary cascade's pool";
          r.asked = want;
          r.free_bytes = ff;
          r.total_bytes = ft;
          r.card = host::select_device().prop.name;
          host::refuse_if_paging(r);
          flat_pool_() = flat_binary_pool_build(lanes, k, kFlatStragCap);
          {
            int il = kFlatIonDefaultLanes;
            int ik = kFlatIonDefaultK;
            if (const char* v = std::getenv("G4GPU_CASCADE_FLAT_ION_LANES")) { il = std::atoi(v); }
            if (const char* v = std::getenv("G4GPU_CASCADE_FLAT_ION_K")) { ik = std::atoi(v); }
            if (il > 0) {
              size_t f2 = 0, t2 = 0;
              cudaMemGetInfo(&f2, &t2);
              host::PagingRequest r2;
              r2.what = "the flat light-ion reaction's pool";
              r2.asked = flat_ion_lane_bytes() * static_cast<std::size_t>(il);
              r2.free_bytes = f2;
              r2.total_bytes = t2;
              r2.card = host::select_device().prop.name;
              host::refuse_if_paging(r2);
              flat_ion_pool_() = flat_ion_pool_build(il, ik, kFlatStragCap);
              if (flat_ion_pool_().ok()) {
                std::printf("flat light-ion reaction: %d lanes of %zu B, %.1f MB, K = %d "
                            "(docs/RISK.md V310)\n",
                            il, flat_ion_lane_bytes(), double(flat_ion_pool_().bytes) / 1048576.0,
                            ik);
              }
            }
          }
          if (flat_pool_().ok()) {
            std::printf("flat Binary cascade: %d lanes of %zu B (a record %zu B), %.1f MB, K = %d "
                        "time steps (docs/RISK.md V308)\n",
                        lanes, flat_binary_lane_bytes(), flat_pool_().record_bytes,
                        double(flat_pool_().bytes) / 1048576.0, k);
          }
        }
      }
    }
    {
      size_t free_b = 0, total_b = 0;
      const cudaError_t mem_err = cudaMemGetInfo(&free_b, &total_b);

      if (batch_ <= 0) {
        if (mem_err != cudaSuccess || free_b == 0) {
          // No way to ask: fall back to something that fits any card this runs on at all.
          batch_ = 262144;
          std::printf("batch: could not read device memory, defaulting to %d events\n", batch_);
        } else {
          // See SetMemoryFraction for why this is a share of free memory rather than all of it.
          // P36 (docs/RISK.md V308): the flat pool's bytes come out of the track buffers' share,
          // so the interaction slots below are sized from the same free memory as before.
          const size_t flat_b = flat_pool_().bytes + flat_ion_pool_().bytes;
          const double share = static_cast<double>(free_b + flat_b) * mem_fraction_;
          const size_t budget =
              (share > static_cast<double>(flat_b)) ? static_cast<size_t>(share) - flat_b : 0;
          long long lo = 1024, hi = 4194304;
          while (lo < hi) {
            const long long mid = lo + (hi - lo + 1) / 2;
            if (track_arena_half_bytes<real_t>(static_cast<long long>(mid * live_per_event_)) * 2 <= budget) {
              lo = mid;
            } else {
              hi = mid - 1;
            }
          }
          batch_ = static_cast<int>(lo);
          std::printf("batch: %d events (auto) - %.2f GB of track buffers, %.0f%% of %.2f GB "
                      "free\n",
                      batch_,
                      double(track_arena_half_bytes<real_t>(static_cast<long long>(batch_ * live_per_event_)) * 2) / 1073741824.0,
                      mem_fraction_ * 100.0, double(free_b) / 1073741824.0);
        }
      } else {
        const size_t want = track_arena_half_bytes<real_t>(static_cast<long long>(batch_ * live_per_event_)) * 2;
        std::printf("batch: %d events (set by the caller) - %.2f GB of track buffers\n", batch_,
                    double(want) / 1073741824.0);
        // REFUSED, not warned about and then attempted.
        //
        // This was a warning until it was tried. The reasoning behind the warning was that an
        // over-large request would fail in cudaMalloc, so announcing it first would give that
        // failure a cause. That is not what happens on Windows: WDDM lets CUDA oversubscribe
        // into host memory, so the allocation SUCCEEDS and the run then crawls, paging tracks
        // over PCIe. A batch 40% too large did not fail - it pinned the GPU until it was
        // killed.
        //
        // So the check has to be the thing that stops it. A caller who names a batch keeps the
        // right to name one; what they do not get is a run that looks like it started. The
        // largest batch that does fit is printed, because the useful response to this message
        // is to pick a number and guessing is a poor use of anyone's time.
        const size_t safe = static_cast<size_t>(static_cast<double>(free_b) * mem_fraction_);
        if (mem_err == cudaSuccess && want > safe) {
          long long lo = 1024, hi = batch_;
          while (lo < hi) {
            const long long mid = lo + (hi - lo + 1) / 2;
            if (track_arena_half_bytes<real_t>(static_cast<long long>(mid * live_per_event_)) * 2 <= safe) {
              lo = mid;
            } else {
              hi = mid - 1;
            }
          }
          std::printf("\nFATAL: a batch of %d needs %.2f GB of track buffers. This run may use\n"
                      "       %.0f%% of the %.2f GB free on this device, which is %.2f GB.\n"
                      "       Refused rather than attempted: CUDA on Windows oversubscribes\n"
                      "       into host memory instead of failing, so the run would appear to\n"
                      "       start and then crawl.\n"
                      "       The largest batch that fits is %lld. Pass 0 to size it\n"
                      "       automatically, or raise SetMemoryFraction if you know what else\n"
                      "       is on the card.\n",
                      batch_, double(want) / 1073741824.0, mem_fraction_ * 100.0,
                      double(free_b) / 1073741824.0, double(safe) / 1073741824.0, lo);
          std::exit(2);
        }
      }
    }

    // One pool per side of the ping-pong, and no division of it at all: a track carries its
    // own species, so storage does not need to know what is in it. Which kernel steps a track
    // is settled later and separately, by the index lists - dispatch, not storage.
    //
    // What that replaced: five capacities fixed before the run - gamma 2 per event, electron 4,
    // positron 1/2, proton 1, alpha 1, doubled for the ping-pong, 17 slots an event in total.
    // They had to be guessed, and measurement showed the guesses wrong in shape as well as
    // size: B1 peaks at 1.00 live gammas an event against 2 reserved and 0.90 electrons against
    // 4, because an electron's range is short and it dies within a step or two while a
    // Compton-scattered gamma keeps flying. Memory set aside for electrons could not be used by
    // gammas however the shower actually went.
    //
    // A pool has no shape to get wrong, so there is nothing here to seed it with.
    {
      // G4GPU_LIVE_PER_EVENT overrides the pool for any program, flag or no flag. It exists to
      // answer one question - is this answer converged in the pool size - which cannot be
      // asked of a program that has no such option, and which is exactly the question a
      // result that moved when the pool changed demands.
      if (const char* env = std::getenv("G4GPU_LIVE_PER_EVENT")) {
        const double v = std::atof(env);
        if (v >= 1.0) { live_per_event_ = v; }
      }
      pool_ = static_cast<long long>(batch_ * live_per_event_);
      // A capacity is an int the whole way down. The automatic sizer cannot ask for more than
      // this - it bisects to at most 4194304 events - but a caller who sets both a batch and a
      // live-track count can, and the arithmetic that follows would truncate rather than fail.
      if (pool_ > 2147483647LL) {
        std::printf("\nFATAL: a pool of %lld live track slots (%d events x %.1f) does not fit\n"
                    "       an int, and a track buffer's capacity is an int the whole way\n"
                    "       down. Lower -batch or -live.\n",
                    pool_, batch_, live_per_event_);
        std::exit(2);
      }
      arena_half_ = track_arena_half_bytes<real_t>(pool_);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_track_arena_, arena_half_ * 2));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_unknown_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_unknown_, 0, sizeof(int)));
      for (int i = 0; i < 2; ++i) {
        TrackSlab slab;
        slab.base = d_track_arena_ + arena_half_ * i;
        slab.capacity = arena_half_;
        // One buffer, the whole pool. No division, so nothing to guess.
        tracks_[i].alloc(static_cast<int>(pool_), &slab);
        if (slab.overflowed) {
          std::printf("\nFATAL: half the track arena measured %zu bytes but the pool carved\n"
                      "       from it did not fit. track_arena_half_bytes and\n"
                      "       allocate_track_buffer have diverged.\n", arena_half_);
          std::exit(2);
        }
        // The dispatch list: ONE array of pool_ ints, bucketed by species each iteration.
        // Sized at the whole pool because the buckets together are the whole pool - which is
        // the point, and is what one array per species could not say: each of those had to be
        // sized at the pool on its own, because any single species may in principle be all of
        // it, so N species cost N times the pool to describe pool-many tracks. Four bytes a
        // slot here, 1.7% on top of a 236-byte track, independent of how many species exist.
        G4GPU_CUDA_CHECK(cudaMalloc(&idx_[i], sizeof(int) * pool_));
        // P33: a count per (species, material, energy) bucket - see count_species (V291).
        G4GPU_CUDA_CHECK(cudaMalloc(&idx_n_[i], sizeof(int) * kNumTrackSpecies * kSortKeys));
      }
      std::printf("track pool: %lld live slots per side (%.1f per event), %.2f GB total\n",
                  pool_, live_per_event_, double(arena_half_ * 2) / 1073741824.0);

      // The drop list, and the pointers into it that every buffer shares. Owned here rather
      // than carved from the slab, because a re-carve would lay track data over it.
      //
      // Sized at a sixteenth of the batch: an overflow that touches more events than that is
      // not an outlier to repair but a pool that is systematically too small, and the batch
      // retry is the right answer to that. The list running past its end is the signal.
      dropped_capacity_ = batch_ / 16 + 1024;
      G4GPU_CUDA_CHECK(cudaMalloc(&d_dropped_, sizeof(int) * dropped_capacity_));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_dropped_count_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_dropped_count_, 0, sizeof(int)));
      for (int i = 0; i < 2; ++i) {
        tracks_[i].view.dropped_event = d_dropped_;
        tracks_[i].view.dropped_count = d_dropped_count_;
        tracks_[i].view.dropped_capacity = dropped_capacity_;
      }
    }
    G4GPU_CUDA_CHECK(cudaMalloc(&d_score_, sizeof(double) * n_scorers_ * batch_));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_status_warn_, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(d_status_warn_, 0, sizeof(int)));

    // ---- the three ledgers for energy and particles that do not become tracks.
    //
    // Allocated unconditionally. `carried_away` is batch_ doubles, the same shape as one row
    // of the score array, so it is not the allocation worth making conditional; the other two
    // are a few dozen words. What would be worth avoiding is a NULL that means "the accounting
    // is off" being indistinguishable from a zero that means "nothing happened", which is why
    // these exist at all - see RunStats.
    G4GPU_CUDA_CHECK(cudaMalloc(&d_carried_away_, sizeof(double) * batch_));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_carried_n_,
                                sizeof(int) * static_cast<int>(ParticleType::kNumTypes)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_refused_,
                                sizeof(int) * static_cast<int>(ParticleType::kNumTypes)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_refused_e_,
                                sizeof(double) * static_cast<int>(ParticleType::kNumTypes)));
    {
      const int nr = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
      G4GPU_CUDA_CHECK(cudaMalloc(&d_had_refused_n_, sizeof(int) * nr));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_had_refused_e_, sizeof(double) * nr));
    }
    G4GPU_CUDA_CHECK(cudaMalloc(&d_killed_energy_, sizeof(double)));
    G4GPU_CUDA_CHECK(cudaMalloc(&d_killed_n_, sizeof(int)));

    // ---- a neutron cross section without a final state is not an allowed state.
    //
    // Refused here rather than discovered on the device. If a table is uploaded without the
    // samplers to go with it, every neutron draws a finite interaction length, arrives at a
    // sub-process branch with nothing to apply, and is killed with its energy dumped locally -
    // a dose that looks like hadronic transport and is a deposit at the first interaction
    // point. The device cannot report that usefully at kernel rates, so the combination is
    // refused where it is set up.
    //
    // P8d LANDED THE FINAL STATES AND THE REFUSAL STAYED, with what it checks turned around.
    // It used to fire on any non-null table, because there were no final states at all; what it
    // now asserts is that the three things `step_neutral` dereferences arrived together. Two of
    // them are cross sections the elastic and capture sub-processes draw a TARGET from
    // (`SampleZandA` on their own data stores, which the general process's table cannot
    // answer), and the third is the level scheme the capture cascade walks.
    //
    // A TABLE WITH NO LEVELS IS THE SUBTLER FAILURE AND IT IS NOT THE OBVIOUS ONE. This comment
    // said the capture "would emit no gamma, so the neutron's binding energy would silently
    // vanish". Measured - `tests/test_neutron_general.cu` section 4, 2000 thermal captures in
    // lead with the table and with a null view - it is the opposite: the cascade takes the
    // CONTINUUM arm of `generate_gamma` instead of walking a discrete scheme and emits
    // **26,485 secondaries carrying 14,609.9 MeV against 9,021 carrying 13,694.7 MeV**, three
    // times the multiplicity and 6.7% more energy. So the failure mode is a plausible capture
    // with a wrong spectrum rather than a missing one, which is exactly the kind of thing that
    // is found in a dose comparison months later. docs/RISK.md V60.
    if (d_neutron_xs_ != nullptr
        && (neutron_tables_.sub.elastic == nullptr || neutron_tables_.sub.capture == nullptr)) {
      std::printf(
          "\nFATAL: the neutron's combined cross-section table is present but a sub-process\n"
          "  data set is not (elastic %s, capture %s). `SampleZandA` draws the target from the\n"
          "  sub-process's OWN data store - G4NeutronGeneralProcess::PostStepDoIt calls\n"
          "  `fCurrentXSS->ComputeCrossSection` before delegating - so the combined table\n"
          "  cannot stand in for it. See physics/hadronic/neutron_wiring.cuh.\n",
          (neutron_tables_.sub.elastic != nullptr) ? "present" : "MISSING",
          (neutron_tables_.sub.capture != nullptr) ? "present" : "MISSING");
      std::exit(2);
    }
    if (d_neutron_xs_ != nullptr && level_tables_.view.n_managers == 0) {
      std::printf(
          "\nFATAL: the neutron's cross sections are on the device and the nuclear level\n"
          "  scheme is not. A capture would still run - and emit the WRONG cascade rather than\n"
          "  none: measured in lead at thermal energy, 2000 captures give 26,485 secondaries\n"
          "  carrying 14,609.9 MeV with a null table against 9,021 carrying 13,694.7 MeV with\n"
          "  the real one, because G4PhotonEvaporation falls back to its continuum arm.\n"
          "  Either let SetNuclearLevelData stay on (the default since P8d) or resolve\n"
          "  G4LEVELGAMMADATA; see host/level_upload.cuh.\n");
      std::exit(2);
    }

    // The secondary arena. One entry per secondary created in a single kernel launch, so it is
    // sized against how many tracks can be in flight rather than against the length of a run.
    //
    // THE DEFAULT IS NOW THE POOL, not half the batch. Half the batch was a measurement of B1 -
    // "about 0.3 secondaries per step" - and a measurement of one physics list is a poor bound
    // for the next one: a hadronic inelastic reaction at a few GeV emits tens of secondaries in
    // one step, and 0.3 per step is not a fact about transport, it is a fact about 6 MeV gammas
    // in water.
    //
    // The pool is a bound rather than an estimate, and it comes from arithmetic that is already
    // in this file. The throttle in BeamOn steps only as many tracks as the output can hold
    // together with their reserved secondaries, so the secondaries one launch can create is at
    // most `cap_out - n_live`, which is at most the pool. An arena of pool_ entries therefore
    // cannot overflow while the output does not - which is the property worth having, because
    // the output's overflow ends the run and the arena's only degrades
    // GetSecondaryInCurrentStep(). Before this the arena was the smaller of the two and could
    // fail quietly first.
    //
    // It costs 8 bytes an entry against a track slot's 236, so pool_ entries is 3.4% on top of
    // the track arena - measured by track_bytes_per_slot, not guessed.
    sec_.capacity = (sec_capacity_ > 0) ? sec_capacity_ : static_cast<int>(pool_);
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.entry, sizeof(unsigned int) * sec_.capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.prev, sizeof(int) * sec_.capacity));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.cursor, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMalloc(&sec_.overflow, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(sec_.cursor, 0, sizeof(int)));
    G4GPU_CUDA_CHECK(cudaMemset(sec_.overflow, 0, sizeof(int)));

    // ---- P15: the interaction queue and the workspace pool it is drained into.
    //
    // THE QUEUE IS A BOUND AND THE POOL IS A CHOICE, and the difference is the whole of
    // docs/RISK.md V188. A launch steps at most `pool_` tracks and one track queues at most one
    // interaction, so a queue of `pool_` entries cannot overflow - the same argument the arena
    // above makes for itself. The POOL is how many of those run at once; a shortage costs
    // launches and not interactions, because the drain loop in BeamOn chunks.
    //
    // THE DEFAULT SLOT COUNT IS 128 AND THE BINDING CONSTRAINT IS NOT THROUGHPUT, IT IS THE
    // DEVICE STACK. At 2,237,728 bytes a slot (P24, V237), 128 slots is 273.2 MB (246.6 MB before
    // P24, 195.9 before P9e's ion arm) - and what is left of an
    // 8 GB card after 86,016 bytes a thread of reserved stack is 0.95 GB, measured (see the
    // `cudaDeviceSetLimit` above and docs/RISK.md V190). 256 slots would take 392 MB of that
    // 0.95 GB and leave the track pool, the queue, the 9.52 MB level scheme and the scene to
    // share what remains. The run prints how deep the queue actually got, so
    // `SetInteractionSlots` can be raised against a measurement rather than a hope - and a
    // shortage costs launches, not interactions.
    n_interaction_slots_ = (interaction_slots_request_ > 0) ? interaction_slots_request_ : 128;
    // P32: `G4GPU_INTERACTION_SLOTS=n` overrides it for any program, for the reason
    // `G4GPU_LIVE_PER_EVENT` exists - examples/B1 has no UI command that reaches the setter, and
    // the question "is the drain bound by its width" cannot be asked of a program that cannot
    // change it. A slot holds no state between interactions that reaches a number (every entry
    // point refills what it reads - V104, V199), so the count moves time and memory, not doses;
    // docs/RISK.md V283 measured that rather than assumed it.
    if (const char* env = std::getenv("G4GPU_INTERACTION_SLOTS")) {
      const int v = std::atoi(env);
      if (v > 0) { n_interaction_slots_ = v; }
    }
    // THE QUEUE'S BOUND IS THE THROTTLE'S, NOT THE POOL'S, and that is a factor of forty-eight.
    //
    // A track queues at most one interaction per launch, so "one entry per pool slot" is a
    // bound - but a loose one, because a launch cannot STEP every live track. `BeamOn`'s
    // throttle lets it step at most `(capacity - live) / max_secondaries_per_step(species)`
    // tracks of a species, and that reservation is 48 for every species that can enqueue. So
    // the number of interactions one launch can produce is at most `pool_ / 48`, and the
    // slack below is for the case where a future species enqueues at a smaller reservation.
    //
    // It is worth having as memory rather than as elegance: `pool_` entries was 368 MB on B1's
    // own batch, out of the 1.17 GB the device stack leaves free (docs/RISK.md V190). This is
    // 8 MB. `kInelasticQueueFull` is still the refusal if the bound is ever wrong, and the
    // stepper's disposal for it is the conservative one.
    const long long q_bound = pool_ / max_secondaries_per_step(kSpeciesProton) + 1024;
    queue_capacity_ = (queue_capacity_request_ > 0)
                          ? queue_capacity_request_
                          : static_cast<int>((q_bound < pool_) ? q_bound : pool_);
    {
      const std::size_t q_bytes =
          sizeof(had::PendingInteraction<real_t>) * static_cast<std::size_t>(queue_capacity_);
      // Refused by name rather than paged (docs/RISK.md V258); the pool's own check is in
      // `had::build_interaction_pool`, which knows its bytes.
      {
        size_t qf = 0, qt = 0;
        cudaMemGetInfo(&qf, &qt);
        host::PagingRequest r;
        r.what = "the interaction queue";
        r.asked = q_bytes + sizeof(int) * (1 + static_cast<std::size_t>(queue_capacity_));
        r.free_bytes = qf;
        r.total_bytes = qt;
        r.card = host::select_device().prop.name;
        host::refuse_if_paging(r);
      }
      G4GPU_CUDA_CHECK(cudaMalloc(&d_queue_, q_bytes));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_queue_cursor_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_queue_cursor_, 0, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_qidx_, sizeof(int) * queue_capacity_));
      G4GPU_CUDA_CHECK(cudaMalloc(
          &d_qcount_,
          sizeof(int) * static_cast<int>(had::InteractionBucket::kNumInteractionBuckets)));
      interaction_bytes_ = q_bytes + sizeof(int) + sizeof(int) * queue_capacity_;

      // The slots, the Binary cascade's channel table, FTFP's pool and P3's Fermi break-up pool,
      // through ONE contract: `had::build_interaction_pool`, declared in
      // physics/hadronic/interaction_pool.cuh and defined beside the models in
      // host/interaction_pool_build.cuh, instantiated by src/host/transport_run_pool.cu. This
      // header reads a size and a device pointer and no model header - docs/RISK.md V233. The
      // one constructed host image per slot (never a memset: V104), the FATAL on an FTFP pool
      // that could not be built, and the bytes the line below prints are the builder's now,
      // unchanged.
      //
      // P32: FTFP's workspaces are a count of their own (`had::ftf_slot_request`, V285): 128,
      // what every slot carried before, or `G4GPU_FTF_SLOTS`, and never more than the slots.
      {
        int ftf_slots = 128;
        if (const char* env = std::getenv("G4GPU_FTF_SLOTS")) {
          const int v = std::atoi(env);
          if (v > 0) { ftf_slots = v; }
        }
        had::ftf_slot_request() = ftf_slots;
      }
      interaction_pool_ = had::build_interaction_pool<real_t>(
          n_interaction_slots_, level_storage_.view(), /*verbose=*/false);
      if (!interaction_pool_.ok) {
        std::printf("\nFATAL: the interaction pool (%d slots) could not be built; see "
                    "host/interaction_pool_build.cuh.\n", n_interaction_slots_);
        std::exit(1);
      }
      // ---- P32: AS MANY SLOTS AS THE CARD HOLDS, NOT 128 (docs/RISK.md V285).
      //
      // The slot count is the drain's WIDTH: a chunk is `n_interaction_slots_` interactions run
      // at once, and the chunks of one iteration run one after the other. V283 measured what 128
      // costs: 96.4% of the 1 GeV proton beam's GPU time is `run_interaction<kBinary>`, 862
      // chunk launches, each one 128 cascades on a handful of SMs of the 80 - and 384 slots ran
      // the same beam 1.5 times faster again than the launch-shape change alone. The constraint
      // the 128 was chosen against is the device stack (the comment above): the raise reserves
      // the interaction kernels' frame for every thread the card can hold, after the pool is
      // built, and a pool that took that memory would have the raise refused by name (V258).
      //
      // So the default is sized here, against what is free NOW minus what the raise WILL ask -
      // read off the five kernels exactly as `RaiseStackForInteractions` reads it - minus a
      // margin for what Upload still allocates (the photo-/lepto-nuclear queue, the voxel
      // scores) and for the driver. The 128-slot pool just built tells us a slot's bytes (the
      // type is opaque here, V233), and when the card holds materially more the pool is rebuilt
      // at that count; capped at 2,048, past which the card cannot keep more single-thread
      // blocks resident (80 SMs x 32 blocks = 2,560 on the Titan V) and the queue's busiest
      // iteration on any beam measured (7,247 on the proton beam) would not fill the chunk
      // anyway. A caller's `SetInteractionSlots` and `G4GPU_INTERACTION_SLOTS` are honoured as
      // given, as before. A slot holds nothing between interactions that reaches a number, so
      // the count moves time and memory and not doses - V283/V285 hold that to the digit.
      const bool slots_by_caller =
          interaction_slots_request_ > 0 || std::getenv("G4GPU_INTERACTION_SLOTS") != nullptr;
      if (!slots_by_caller) {
        // A slot's bytes and nothing for FTFP: its workspaces stay at the count set below.
        const std::size_t per_slot = interaction_pool_.slot_bytes;
        cudaFuncAttributes ka[5] = {};
        cudaFuncGetAttributes(&ka[0],
                              run_interaction<real_t, had::InteractionBucket::kFtfp, StepHook>);
        cudaFuncGetAttributes(&ka[1],
                              run_interaction<real_t, had::InteractionBucket::kBertini, StepHook>);
        cudaFuncGetAttributes(&ka[2],
                              run_interaction<real_t, had::InteractionBucket::kBinary, StepHook>);
        cudaFuncGetAttributes(&ka[3],
                              run_interaction<real_t, had::InteractionBucket::kLightIon, StepHook>);
        cudaFuncGetAttributes(&ka[4],
                              run_interaction<real_t, had::InteractionBucket::kAtRest, StepHook>);
        std::size_t need = 0;
        for (const cudaFuncAttributes& a : ka) { need = std::max(need, a.localSizeBytes); }
        std::size_t want = ((need + 4095u) / 4096u) * 4096u + 4096u;
        want = std::max(want, stepping_stack_bytes());
        std::size_t have = 0;
        cudaDeviceGetLimit(&have, cudaLimitStackSize);
        const host::SelectedDevice& dev = host::select_device();
        const std::size_t threads = static_cast<std::size_t>(dev.prop.multiProcessorCount) *
                                    static_cast<std::size_t>(dev.prop.maxThreadsPerMultiProcessor);
        const std::size_t raise = (want > have) ? (want - have) * threads : 0;
        constexpr std::size_t kMargin = std::size_t(512) << 20;
        constexpr int kMaxAutoSlots = 2048;
        size_t pf = 0, pt = 0;
        cudaMemGetInfo(&pf, &pt);
        const std::size_t reserve = raise + kMargin;
        const long long extra =
            (pf > reserve && per_slot > 0) ? static_cast<long long>((pf - reserve) / per_slot) : 0;
        const int target = static_cast<int>(
            std::min<long long>(kMaxAutoSlots, n_interaction_slots_ + extra));
        if (target >= n_interaction_slots_ + n_interaction_slots_ / 4) {
          had::free_interaction_pool<real_t>(interaction_pool_);
          n_interaction_slots_ = target;
          interaction_pool_ = had::build_interaction_pool<real_t>(
              n_interaction_slots_, level_storage_.view(), /*verbose=*/false);
          if (!interaction_pool_.ok) {
            std::printf("\nFATAL: the interaction pool (%d slots) could not be built; see "
                        "host/interaction_pool_build.cuh.\n", n_interaction_slots_);
            std::exit(1);
          }
        }
        std::printf("interaction slots: %d, sized to the card - %.2f GB free before the pool, "
                    "%.2f GB kept for the stack raise (%zu B a thread x %zu threads) and %.2f GB "
                    "of margin (docs/RISK.md V285)\n",
                    n_interaction_slots_, double(pf) / 1073741824.0, double(raise) / 1073741824.0,
                    want, threads, double(kMargin) / 1073741824.0);
      }
      interaction_bytes_ += interaction_pool_.bytes;

      std::printf("interaction pool: %d slots of %zu B + FTFP's %zu B on %d of them = %.1f MB; "
                  "queue %d entries of %zu B = %.1f MB\n",
                  n_interaction_slots_, interaction_pool_.slot_bytes,
                  interaction_pool_.ftf_workspace_bytes, interaction_pool_.n_ftf_slots,
                  double(interaction_pool_.slots_bytes + interaction_pool_.ftf_bytes) / 1048576.0,
                  queue_capacity_, sizeof(had::PendingInteraction<real_t>),
                  double(q_bytes) / 1048576.0);
    }

    // ---- P19: the photo- and lepto-nuclear queue, and its bound.
    //
    // A SECOND QUEUE AND NOT THE FIRST ONE ENLARGED: it is drained by `run_emextra_drain`, which
    // reads its cursor on the device, so a photon run pays no host synchronisation for it (see
    // that kernel's header).
    //
    // ITS CAPACITY IS A MARGIN, NOT P15's KIND OF BOUND, AND THE PROOF WAS MEASURED OUT OF IT.
    // The proof exists: a photon or an e+- is stepped against a reservation of FOUR
    // (`max_secondaries_per_step`), so one launch steps at most `pool_/4` of them, plus
    // `pool_/48` muons, and each queues at most once. The first version allocated exactly that,
    // and on B1's gamma batch (1,834,944 events, 4 live) it was 1,988,880 entries - 743.5 MB -
    // and on a hadron beam's 32 live an event it would have been 6.2 GB. With it, the 6 MeV
    // gate's one photo-nuclear reaction raised the device stack into a card with 0.00 GB left
    // and the event loop went from 1.2 s to 13.5 s (docs/RISK.md V210).
    //
    // What a launch actually queues is a Bernoulli sum: a photon's photo-nuclear probability is,
    // since P21, `1 - P9` of the general process's table 9 in zone 2 - 6.09% at its largest in
    // B1's four materials, at the table's node at 23.07 MeV in air (5.08% in water, 4.43% in
    // compact bone) - where P19 had sigN/(total + sigN), 4.27% at its largest (compact bone,
    // 22 MeV); and a lepton's is about 1e-5 a step. So a tenth of the EM budget bounds the
    // expectation 1.6 times over, the Chernoff tail past it is exp(-0.0095 N) for a launch of N
    // photons all at that peak - below 1e-13 from N = 3,200 - and the 1,024-entry floor covers
    // smaller launches many times over. It is capped at 131,072 entries (51.4 MB), which a
    // launch reaches only by stepping two million photons at the resonance peak at once.
    // `kInelasticQueueFull` + the SIZE row is still the tripwire, with the conservative
    // disposal, and the run prints how deep the queue got.
    {
      constexpr long long kEmxQueueCap = 131072;
      const long long e_bound = pool_ / (10 * max_secondaries_per_step(kSpeciesGamma)) + 1024;
      long long e_cap = (e_bound < kEmxQueueCap) ? e_bound : kEmxQueueCap;
      if (e_cap > pool_) { e_cap = pool_; }
      emx_queue_capacity_ = static_cast<int>(e_cap);
      const std::size_t e_bytes =
          sizeof(had::PendingInteraction<real_t>) * static_cast<std::size_t>(emx_queue_capacity_);
      {
        size_t ef = 0, et = 0;
        cudaMemGetInfo(&ef, &et);
        host::PagingRequest r;
        r.what = "the photo-/lepto-nuclear queue";
        r.asked = e_bytes + sizeof(int);
        r.free_bytes = ef;
        r.total_bytes = et;
        r.card = host::select_device().prop.name;
        host::refuse_if_paging(r);  // docs/RISK.md V258
      }
      G4GPU_CUDA_CHECK(cudaMalloc(&d_emx_queue_, e_bytes));
      G4GPU_CUDA_CHECK(cudaMalloc(&d_emx_cursor_, sizeof(int)));
      G4GPU_CUDA_CHECK(cudaMemset(d_emx_cursor_, 0, sizeof(int)));
      G4GPU_CUDA_CHECK(
          cudaMalloc(&d_emx_stats_, sizeof(unsigned long long) * kEmxCounters));
      G4GPU_CUDA_CHECK(
          cudaMemset(d_emx_stats_, 0, sizeof(unsigned long long) * kEmxCounters));
      interaction_bytes_ += e_bytes + sizeof(int);
      // THE TWO DRAIN KERNELS' FRAMES, READ OFF THE KERNELS AT UPLOAD. They launch on photon and
      // electron runs that never raise the device stack (`RaiseStackForInteractions` waits for a
      // queued interaction), so a frame past the stepping kernels' limit would make the driver
      // raise it for
      // the whole card at the first launch - silently, V196 - and cost the gamma gate what V190
      // measured. The line says which, and by how much, if it ever does.
      cudaFuncAttributes fa_photo{}, fa_lepto{};
      G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
          &fa_photo,
          run_emextra_drain<real_t, had::InteractionBucket::kPhotoNuclear, StepHook>));
      G4GPU_CUDA_CHECK(cudaFuncGetAttributes(
          &fa_lepto,
          run_emextra_drain<real_t, had::InteractionBucket::kLeptoNuclear, StepHook>));
      emx_frame_photo_ = fa_photo.localSizeBytes;
      emx_frame_lepto_ = fa_lepto.localSizeBytes;
      const std::size_t limit = stepping_stack_bytes();
      std::printf("photo-/lepto-nuclear queue: %d entries = %.1f MB; drain frames %zu B "
                  "(photonNuclear) and %zu B (lepto-nuclear) against the %zu B stack%s\n",
                  emx_queue_capacity_, double(e_bytes) / 1048576.0, emx_frame_photo_,
                  emx_frame_lepto_, limit,
                  (emx_frame_photo_ > limit || emx_frame_lepto_ > limit)
                      ? " - ABOVE IT: the driver will raise the device stack at the first "
                        "drain launch (docs/RISK.md V196, V210)"
                      : "");
    }

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


// ---------------------------------------------------------------- P32: the loop's own profile
//
// `G4GPU_LOOP_PROFILE=1` makes BeamOn time every launch class with CUDA events, count the
// host-visible launches and the synchronising copies of every iteration, and bin the iterations
// by how full the pool was - and print one table at the end of the run. It exists because the
// profilers on the gating machine predate its driver (Nsight Systems 2021.5.2 and Nsight Compute
// 2022.1.0 against CUDA 12.9 - docs/RISK.md V283 says what they did), and because the question
// the performance phase has to answer first is not "which kernel is slow" but "where does an
// iteration's time go, and how much of it is spent stepping a handful of survivors".
//
// IT CHANGES NO NUMBER. It records events on the stream every launch is already on and reads
// them back once per iteration, after the iteration's last launch; nothing it does reaches a
// kernel argument, a track or a score. What it does change is the host's timing - one more
// synchronisation an iteration - so a profiled run's wall clock is not a rate: rates are read
// off runs with it off. Off, it costs one well-predicted branch per launch site.
//
// `G4GPU_LOOP_PROFILE_CSV=<path>` also writes one row per iteration (the live count, the steps
// taken, the wall and GPU milliseconds), which is the "live tracks against the iteration
// number" curve.
struct LoopProfile {
  // The launch classes: the counting sort, one per stepping species, the interaction queue's
  // read-back and binning, one per interaction bucket, the two photo-/lepto-nuclear drains.
  static constexpr int kSort = 0;
  static constexpr int kStepBase = 1;
  static constexpr int kQueueBin = kStepBase + kNumTrackSpecies;
  static constexpr int kIntBase = kQueueBin + 1;
  static constexpr int kNumBuckets = 5;  // kFtfp, kBertini, kBinary, kLightIon, kAtRest
  static constexpr int kEmxBase = kIntBase + kNumBuckets;
  // P37: the species' stepping launches, timed as one span when they run on their own streams
  // (each species' own class then counts its launches and tracks, and holds no time).
  static constexpr int kStepStreams = kEmxBase + 2;
  static constexpr int kNumClasses = kEmxBase + 3;
  // The live-count bins, as a fraction of the pool: < 1%, 1-10%, 10-50%, >= 50%.
  static constexpr int kNumBins = 4;

  bool on = false;
  std::FILE* csv = nullptr;
  std::vector<cudaEvent_t> ev;
  std::vector<int> span_class;  // one per begin/end pair recorded this iteration
  int used = 0;
  int pending = -1;
  double ms[kNumClasses] = {};
  long long calls[kNumClasses] = {};
  long long items[kNumClasses] = {};  // tracks stepped, or interactions run, by the class
  long long launches = 0;             // kernel launches the host issued
  long long syncs = 0;                // synchronising device-to-host copies
  long long memsets = 0;              // memsets and host-to-device copies
  long long iterations = 0;
  long long it_bin[kNumBins] = {};
  long long steps_bin[kNumBins] = {};
  double wall_bin[kNumBins] = {};
  double gpu_bin[kNumBins] = {};
  double wall_total = 0.0, gpu_total = 0.0;
  // This iteration's.
  long long it_launches = 0, it_syncs = 0, it_memsets = 0;
  long long max_it_launches = 0, max_it_syncs = 0;
  std::chrono::steady_clock::time_point t0;

  void init() {
    const char* env = std::getenv("G4GPU_LOOP_PROFILE");
    on = (env != nullptr && env[0] != '\0' && env[0] != '0');
    if (!on) { return; }
    ev.resize(2 * 4 * kNumClasses);
    for (cudaEvent_t& e : ev) { G4GPU_CUDA_CHECK(cudaEventCreate(&e)); }
    if (const char* path = std::getenv("G4GPU_LOOP_PROFILE_CSV")) {
      csv = std::fopen(path, "w");
      if (csv != nullptr) {
        std::fprintf(csv, "iteration,n_live,pool,steps,wall_ms,gpu_ms,step_ms,int_ms,"
                          "launches,syncs\n");
      }
    }
  }
  void release() {
    for (cudaEvent_t& e : ev) { cudaEventDestroy(e); }
    ev.clear();
    if (csv != nullptr) { std::fclose(csv); csv = nullptr; }
  }
  void start_iteration() {
    if (!on) { return; }
    used = 0;
    pending = -1;
    span_class.clear();
    it_launches = it_syncs = it_memsets = 0;
    t0 = std::chrono::steady_clock::now();
  }
  void begin(int c) {
    if (!on || used + 2 > static_cast<int>(ev.size())) { return; }
    G4GPU_CUDA_CHECK(cudaEventRecord(ev[used]));
    pending = c;
  }
  void end(int c, long long n_items) {
    // A span whose begin() was refused for want of events is dropped, not mis-paired.
    if (!on || pending != c) { return; }
    G4GPU_CUDA_CHECK(cudaEventRecord(ev[used + 1]));
    span_class.push_back(c);
    used += 2;
    pending = -1;
    ++calls[c];
    items[c] += n_items;
  }
  /// P37: a launch counted in its class without a span of its own (see kStepStreams).
  void count_only(int c, long long n_items) {
    if (!on) { return; }
    ++calls[c];
    items[c] += n_items;
  }
  void launch(int n = 1) { if (on) { launches += n; it_launches += n; } }
  void sync(int n = 1) { if (on) { syncs += n; it_syncs += n; } }
  void memop(int n = 1) { if (on) { memsets += n; it_memsets += n; } }
  /// Closes the iteration: waits for its last event, sums every span into its class, and bins
  /// the iteration by the live count against the pool.
  void close_iteration(long long iteration, int n_live, int pool, long long steps) {
    if (!on) { return; }
    double gpu = 0.0, step_ms = 0.0, int_ms = 0.0;
    if (used > 0) {
      G4GPU_CUDA_CHECK(cudaEventSynchronize(ev[used - 1]));
      for (int s = 0; s * 2 + 1 < used; ++s) {
        float m = 0.0f;
        G4GPU_CUDA_CHECK(cudaEventElapsedTime(&m, ev[2 * s], ev[2 * s + 1]));
        const int c = span_class[s];
        ms[c] += m;
        gpu += m;
        if ((c >= kStepBase && c < kQueueBin) || c == kStepStreams) { step_ms += m; }
        if (c >= kIntBase && c < kEmxBase) { int_ms += m; }
      }
    } else {
      G4GPU_CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double wall =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double f = (pool > 0) ? static_cast<double>(n_live) / pool : 0.0;
    const int b = (f < 0.01) ? 0 : (f < 0.10) ? 1 : (f < 0.50) ? 2 : 3;
    ++it_bin[b];
    steps_bin[b] += steps;
    wall_bin[b] += wall;
    gpu_bin[b] += gpu;
    wall_total += wall;
    gpu_total += gpu;
    ++iterations;
    max_it_launches = std::max(max_it_launches, it_launches);
    max_it_syncs = std::max(max_it_syncs, it_syncs);
    if (csv != nullptr) {
      std::fprintf(csv, "%lld,%d,%d,%lld,%.4f,%.4f,%.4f,%.4f,%lld,%lld\n", iteration, n_live,
                   pool, steps, wall, gpu, step_ms, int_ms, it_launches, it_syncs);
    }
  }
  static const char* class_name(int c) {
    static char buf[kNumClasses][48];
    if (c == kSort) { return "count/scatter species"; }
    if (c == kQueueBin) { return "queue read-back + binning"; }
    if (c >= kStepBase && c < kQueueBin) {
      const int sp = c - kStepBase;
      if (sp == kSpeciesStrangeCharged) { return "step strange charged"; }
      if (sp == kSpeciesStrangeNeutral) { return "step strange neutral"; }
      std::snprintf(buf[c], sizeof(buf[c]), "step %s", particle_name(species_of_index(sp)));
      return buf[c];
    }
    if (c >= kIntBase && c < kEmxBase) {
      static const char* const kB[kNumBuckets] = {"interaction kFtfp", "interaction kBertini",
                                                   "interaction kBinary", "interaction kLightIon",
                                                   "interaction kAtRest"};
      return kB[c - kIntBase];
    }
    if (c == kStepStreams) { return "stepping, species at once"; }
    if (c == kEmxBase) { return "drain kPhotoNuclear"; }
    return "drain kLeptoNuclear";
  }
  void report(double run_ms) const {
    if (!on || iterations == 0) { return; }
    std::printf("\n== loop profile (G4GPU_LOOP_PROFILE, docs/RISK.md V283) ==\n");
    std::printf("  %lld iterations; wall %.1f ms in the loop (%.1f ms of GPU time in the timed "
                "launches, %.1f ms between them); the run's own clock %.1f ms\n",
                iterations, wall_total, gpu_total, wall_total - gpu_total, run_ms);
    std::printf("  per iteration: %.1f kernel launches (max %lld), %.2f synchronising read-backs "
                "(max %lld), %.1f memsets/uploads\n",
                double(launches) / iterations, max_it_launches, double(syncs) / iterations,
                max_it_syncs, double(memsets) / iterations);
    std::printf("  %-28s %10s %12s %14s %8s %12s\n", "class", "spans", "ms", "items",
                "% GPU", "us/span");
    for (int c = 0; c < kNumClasses; ++c) {
      if (calls[c] == 0) { continue; }
      std::printf("  %-28s %10lld %12.1f %14lld %7.1f%% %12.1f\n", class_name(c), calls[c],
                  ms[c], items[c], (gpu_total > 0) ? 100.0 * ms[c] / gpu_total : 0.0,
                  1000.0 * ms[c] / calls[c]);
    }
    static const char* const kBin[kNumBins] = {"< 1% of the pool", "1-10%", "10-50%", ">= 50%"};
    std::printf("  %-20s %10s %8s %14s %12s %12s\n", "live tracks", "iterations", "% its",
                "steps", "wall ms", "GPU ms");
    for (int b = 0; b < kNumBins; ++b) {
      std::printf("  %-20s %10lld %7.1f%% %14lld %12.1f %12.1f\n", kBin[b], it_bin[b],
                  100.0 * it_bin[b] / iterations, steps_bin[b], wall_bin[b], gpu_bin[b]);
    }
  }
};

template <typename real_t, typename StepHook>
RunStats TransportEngine<real_t, StepHook>::BeamOn(int n_events, const Primary<real_t>* primaries,
                                         unsigned int seed,
                                         std::vector<double>& score_sum,
                                         std::vector<double>& score_sum_sq,
                                         vis::TrajectoryBuffer traj, EventSink* sink,
                                         long long stream_pos) {
    score_sum.assign(n_scorers_, 0.0);
    score_sum_sq.assign(n_scorers_, 0.0);
    // Zeroed per run, not per batch: this accumulates the whole run's deposit per cell.
    if (d_voxel_score_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_voxel_score_, 0, sizeof(double) * n_voxel_cells_));
    }
    RunStats st;
    if (d_status_warn_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_status_warn_, 0, sizeof(int)));
    }
    if (sec_.overflow != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(sec_.overflow, 0, sizeof(int)));
    }
    // Per run, not per batch: these are counts and totals over the whole run, unlike the
    // per-event carried-away array which is batch-local and zeroed with the scores.
    const size_t kTypeBytes = sizeof(int) * static_cast<size_t>(ParticleType::kNumTypes);
    if (d_carried_n_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_carried_n_, 0, kTypeBytes));
    }
    if (d_refused_ != nullptr) { G4GPU_CUDA_CHECK(cudaMemset(d_refused_, 0, kTypeBytes)); }
    if (d_refused_e_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_refused_e_, 0,
                                  sizeof(double) * static_cast<size_t>(ParticleType::kNumTypes)));
    }
    {
      const size_t nr = static_cast<size_t>(had::HadronicRefusal::kNumHadronicRefusals);
      if (d_had_refused_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_had_refused_n_, 0, sizeof(int) * nr));
      }
      if (d_had_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_had_refused_e_, 0, sizeof(double) * nr));
      }
    }
    if (d_killed_energy_ != nullptr) {
      G4GPU_CUDA_CHECK(cudaMemset(d_killed_energy_, 0, sizeof(double)));
    }
    if (d_killed_n_ != nullptr) { G4GPU_CUDA_CHECK(cudaMemset(d_killed_n_, 0, sizeof(int))); }
    // P19's counters, per run like the ledgers beside them.
    if (d_emx_stats_ != nullptr) {
      G4GPU_CUDA_CHECK(
          cudaMemset(d_emx_stats_, 0, sizeof(unsigned long long) * kEmxCounters));
    }
    iterations_run_ = 0;
    hadron_iterations_ = 0;
    neutron_iterations_ = 0;
    if (primaries == nullptr || n_events <= 0) { return st; }

    // Every primary is checked against what a kernel can step. A primary with no kernel used
    // to land in the gamma buffer and be transported as a gamma - a run that produced a
    // plausible dose for physics that never happened - and the guard that replaced that was a
    // hand-written list of five species. It is species_disposition now, so the list cannot
    // drift from the set of kernels the way prose does.
    //
    // Checked over every primary rather than once for the source, because a generator is free
    // to fire a different species per event and the guard has to cover what it actually
    // produced. It is one switch per event against a whole shower.
    for (int i = 0; i < n_events; ++i) {
      const ParticleType p = primaries[i].particle;
      const SpeciesDisposition disp = species_disposition(p);
      if (disp == SpeciesDisposition::kStepped) {
        // P30: A REAL NUCLEUS NEEDS ITS NUCLIDE. `kGenericIon` is a species with no particle of
        // its own - `particle_def(kGenericIon)` is G4GenericIon's 938.2723 MeV placeholder - so
        // a primary of it is only a particle when `ion_za` names one, with a mass Geant4 gives
        // it (G4NucleiProperties::GetNuclearMass > 0). Without that `step_hadron` would refuse
        // every step as `kIonWithoutNuclide` and deposit the energy where it stood; refusing
        // the run says so once, by name, before anything is launched. The G4 layer's gun cannot
        // build such a primary (`/gun/ion` refuses it, G4RunManager::RunEvents too); this is
        // the guard for a caller that fills `Primary` records itself.
        if (p == ParticleType::kGenericIon
            && (primaries[i].ion_za == 0
                || !data::nuclear_mass_known(ion_a_of(primaries[i].ion_za),
                                             ion_z_of(primaries[i].ion_za)))) {
          std::printf(
              "\nFATAL: a \"GenericIon\" primary with no nuclide (event %d, ion_za %u).\n"
              "  GenericIon is the placeholder whose tables every real nucleus reads; a primary\n"
              "  nucleus is `/gun/particle ion` + `/gun/ion Z A` (Primary::ion_za), and its\n"
              "  (Z, A) must have a nuclear mass. Refusing rather than stepping a nucleus of\n"
              "  unknown mass.\n",
              i, static_cast<unsigned int>(primaries[i].ion_za));
          std::exit(2);
        }
        // Every charged hadron is stepped from the hadron dE/dx and range tables. A scene
        // uploaded without them (b1_gpu_sched builds a gamma-only Scene deliberately) would
        // step the primary once, find a null table, and drop it. That is a silent zero, so it
        // is refused here.
        //
        // `is_heavy_charged` and not "everything but the three EM species". A neutron and a
        // pi0 are stepped by step_neutral, which has no continuous loss and therefore never
        // reads the range table - so refusing them for want of it would refuse the one run
        // that is worth making today: a neutron primary with no cross section, streaming to
        // the world boundary. The predicate names the species that read the table rather than
        // the species that do not.
        if (scene_.hadron_range == nullptr && is_heavy_charged(p)) {
          std::printf(
              "\nFATAL: a \"%s\" primary was given to a Scene with no hadron range table\n"
              "  (event %d). step_hadron cannot advance a track without one, and dropping it\n"
              "  would look like a run that simply deposited nothing.\n",
              particle_name(p), i);
          std::exit(2);
        }
        continue;
      }
      if (disp == SpeciesDisposition::kCounted) {
        // A neutrino primary is not an error and not transportable either: its whole energy
        // leaves the event at the vertex. Refused rather than silently booked as escaped,
        // because a run of neutrino primaries would report a dose of exactly zero and that is
        // a number somebody would take at face value. QBBC would transport it - across the
        // world in one step, depositing nothing - so the answer is the same and saying so is
        // better than producing it.
        std::printf(
            "\nFATAL: \"%s\" is not a primary this transport will fire (event %d).\n"
            "  A neutrino has no process in QBBC, so every event would deposit exactly zero\n"
            "  and the run would look like a result. Its energy is booked as carried away\n"
            "  when a decay makes one; as a primary there would be nothing else in the event.\n",
            particle_name(p), i);
        std::exit(2);
      }
      std::printf(
          "\nFATAL: \"%s\" (species %d) is not transported (event %d).\n"
          "  See G4RunManager::CheckSpecies for the current species set and what each\n"
          "  missing one needs. Refusing rather than transporting it as something it is not.\n",
          particle_name(p), static_cast<int>(p), i);
      std::exit(2);
    }

    // The primaries, uploaded a batch at a time. Sized once for the batch rather than for the
    // run: a ten-million-event run would otherwise want half a gigabyte resident for
    // something the device reads once.
    if (d_primaries_ == nullptr) {
      G4GPU_CUDA_CHECK(cudaMalloc(&d_primaries_, sizeof(Primary<real_t>) * batch_));
    }

    std::vector<double> ev_scores(static_cast<size_t>(n_scorers_), 0.0);

    // ---- P32: THE SCORE WINDOW, AND THE BATCH REFILL IT MAKES POSSIBLE (docs/RISK.md V284).
    //
    // Until P32 a batch was both things at once: the events whose primaries were seeded together
    // AND the columns of the per-event score array. So a run of more events than a batch drained
    // its first batch to the last track - the batch's time set by its longest shower, the card
    // stepping a handful of survivors for the tail - before the next batch's primaries could be
    // seeded, because the next batch needed the score columns the survivors' events still held.
    //
    // The two are separate now. The WINDOW is the score array's width: up to `win_cap` events
    // whose per-event scores and carried-away energy are kept on the device and read back
    // together. The BATCH is how many events are in flight: the window's first `batch_` events
    // are seeded at the first iteration, as before, and every later iteration seeds the next ones
    // into whatever the pool has freed - as many as bring the live count back to `batch_` - until
    // the window's last primary is in. Only then does the drain run down its tail, once per
    // window and not once per batch.
    //
    // NOTHING ABOUT A TRACK CHANGES. Its key is `seed ^ (key_base_for(stream_pos, window base) +
    // window index)`, which is `key_base_for(stream_pos, run index)` exactly as the batch's was,
    // and every random number it and its descendants draw is keyed on that and on its own step
    // count - never on the iteration it was seeded in or the tracks beside it. Each event's score
    // is its own column, read back and summed over events in ascending order as before. So the
    // dose is bit-identical, and V284 holds it to the digit on the gates and on runs of several
    // batches forced with G4GPU_BATCH.
    //
    // A run of no more events than a batch - every gate here, and every B1 beam at the default
    // batch - seeds all of them at the first iteration and never refills: the schedule is
    // exactly the one it was. `G4GPU_REFILL=0` turns the refill off for any run (each window is
    // then seeded whole, in batches as before), so the comparison can be made.
    //
    // THE WINDOW'S MEMORY is per event: a score per scorer, the carried-away double and the
    // primary itself. It is capped at 256 MB and at an eighth of what the card has free, and
    // never below the batch, so a window costs what the batch did on a run that fits one, and a
    // ten-million-event run gets windows of a few million rather than an allocation it cannot
    // hold. Allocated here, for this BeamOn, and freed at its end; the batch-sized arrays Upload
    // made are used as they were when the window is no wider than the batch.
    bool refill = true;
    if (const char* env = std::getenv("G4GPU_REFILL")) {
      refill = !(env[0] == '0' && env[1] == '\0');
    }
    int win_cap = batch_;
    double* d_win_score = d_score_;
    double* d_win_carried = d_carried_away_;
    Primary<real_t>* d_win_prim = d_primaries_;
    bool win_owned = false;
    if (refill && n_events > batch_) {
      const std::size_t per_event = sizeof(double) * static_cast<std::size_t>(n_scorers_)
                                    + ((d_carried_away_ != nullptr) ? sizeof(double) : 0)
                                    + sizeof(Primary<real_t>);
      size_t wf = 0, wt = 0;
      cudaMemGetInfo(&wf, &wt);
      const std::size_t budget = std::min<std::size_t>(std::size_t(256) << 20, wf / 8);
      const long long fit = static_cast<long long>(budget / per_event);
      const long long want = std::min<long long>(n_events, std::max<long long>(fit, batch_));
      if (want > batch_) {
        win_cap = static_cast<int>(want);
        G4GPU_CUDA_CHECK(cudaMalloc(&d_win_score,
                                    sizeof(double) * static_cast<std::size_t>(n_scorers_) * win_cap));
        if (d_carried_away_ != nullptr) {
          G4GPU_CUDA_CHECK(cudaMalloc(&d_win_carried, sizeof(double) * win_cap));
        }
        G4GPU_CUDA_CHECK(cudaMalloc(&d_win_prim, sizeof(Primary<real_t>) * win_cap));
        win_owned = true;
      }
    }
    // The score array's stride - its column count - is the window's, and every kernel indexes
    // `score[slot * stride + event]` with it (the kernels' `batch` argument has only ever been
    // that stride).
    const int stride = win_cap;
    long long refills = 0, refilled_events = 0;

    std::vector<double> h_score(static_cast<size_t>(n_scorers_) * stride);
    // P32: off unless G4GPU_LOOP_PROFILE asks; see LoopProfile.
    LoopProfile prof;
    prof.init();
    // P37 (V304): the counting sort's 40-integer summary, on the card and in pinned host memory.
    // A failed allocation leaves both null and the sort takes the host's prefix sum.
    // The host side is MAPPED: the kernels write it themselves, and the loop's other two
    // read-backs (the pool's append cursor, the interaction queue's) land in its last slots.
    constexpr int kMapCount = 2 * kNumTrackSpecies;      // the pool's count
    constexpr int kMapQueued = 2 * kNumTrackSpecies + 1; // the interaction queue's cursor
    int* d_sort_summary = nullptr;
    int* h_sort_summary = nullptr;
    int* d_map = nullptr;
    if (cudaMalloc(&d_sort_summary, sizeof(int) * 2 * kNumTrackSpecies) != cudaSuccess
        || cudaHostAlloc(&h_sort_summary, sizeof(int) * (2 * kNumTrackSpecies + 2),
                         cudaHostAllocMapped) != cudaSuccess
        || cudaHostGetDevicePointer(&d_map, h_sort_summary, 0) != cudaSuccess) {
      cudaGetLastError();
      if (d_sort_summary != nullptr) { cudaFree(d_sort_summary); }
      if (h_sort_summary != nullptr) { cudaFreeHost(h_sort_summary); }
      d_sort_summary = nullptr;
      h_sort_summary = nullptr;
      d_map = nullptr;
    }
    // A device integer read back: through the mapped slot when there is one, else as before.
    auto read_back = [&](const int* dsrc, int slot) -> int {
      if (d_map == nullptr) {
        int c = 0;
        G4GPU_CUDA_CHECK(cudaMemcpy(&c, dsrc, sizeof(int), cudaMemcpyDeviceToHost));
        return c;
      }
      copy_int_to_host<<<1, 32>>>(dsrc, d_map + slot);
      G4GPU_CUDA_CHECK(cudaGetLastError());
      G4GPU_CUDA_CHECK(cudaStreamSynchronize(0));
      return reinterpret_cast<volatile int*>(h_sort_summary)[slot];
    };
    // TrackBuffer::count's clamp, on the mapped read.
    auto live_count = [&](const DeviceTracks<real_t>& b) -> int {
      const int c = read_back(b.view.count, kMapCount);
      return (c > b.view.capacity) ? b.view.capacity : c;
    };

    // ---- P34: THE DEFERRED DRAIN (docs/RISK.md V294).
    //
    // An iteration that queues interactions used to drain them before any track took its next
    // step, so every live track waited for the slowest reaction of every such iteration. On the
    // Fe56 beam that is the whole run: 5,098 of 8,433 iterations drain about twelve reactions
    // each, at 102 ms of interaction against 4 ms of stepping (V293). P32 tried to hide it on a
    // second stream (V286); V294 measured that this card's driver runs the two streams one after
    // the other - a 0.2 ms launch on the main stream waits for everything already queued on the
    // other, one kernel or fifty, legacy stream or not - so what P32's 1.35x was made of is the
    // BATCHING its ring did, not overlap. This is that batching, on one stream and on a schedule
    // the host decides from counts alone: an iteration's queued interactions are copied into a
    // pending ring instead of drained, and the ring is drained when it holds a full-width chunk
    // (`n_interaction_slots_` entries), when its oldest entry has waited `defer` iterations, when
    // it would overflow, and whenever nothing else is alive. A pending track just takes its next
    // step later; its key, its step count and therefore every number it draws are its own
    // (V284), and V294 holds the four beams and the eleven gates to the digit with it on.
    //
    // `G4GPU_DEFER_ITERATIONS=k` sets the age bound (0 drains every iteration, as before).
    // Each pending entry keeps its 48-slot output reservation (`max_secondaries_per_step`) in
    // the throttle below until it is drained, so the deferral does not spend the room the
    // stepping kernels' outputs need.
    int defer = kDeferIterations;
    if (const char* env = std::getenv("G4GPU_DEFER_ITERATIONS")) {
      defer = std::max(0, std::atoi(env));
    }
    had::PendingInteraction<real_t>* d_pending = nullptr;
    int pending_n = 0, pending_from = 0, pending_cap = 0;
    long long deferred_drains = 0, deferred_entries = 0;
    if (defer > 0 && d_queue_ != nullptr && interaction_pool_.ok) {
      pending_cap = std::min(kPendingCap, queue_capacity_);
      G4GPU_CUDA_CHECK(
          cudaMalloc(&d_pending, sizeof(had::PendingInteraction<real_t>) * pending_cap));
    }
    // The last iteration's wiring and books, for a drain at the top of an iteration that has
    // not built its own yet; they are the run's and do not change between iterations.
    had::HadronicWiring<real_t> last_wiring{};
    EmitterBooks last_books{};
    // One drain of `n` queued entries at `items` into `out`: bin by model (histogram, host
    // prefix sum, scatter), then each bucket in chunks of the slot count - P15's drain, with
    // the queue as a parameter so the pending ring can be drained the same way.
    auto drain_interactions = [&](const had::PendingInteraction<real_t>* items, int n,
                                  TrackBuffer<real_t> out,
                                  const had::HadronicWiring<real_t>& wiring,
                                  const EmitterBooks& bk) {
      if (n <= 0) { return; }
      RaiseStackForInteractions();  // once per run; see the note above `any_hadron`'s use
      constexpr int kNB = static_cast<int>(had::InteractionBucket::kNumInteractionBuckets);
      prof.begin(LoopProfile::kQueueBin);
      const int qblocks = (n + threads_ - 1) / threads_;
      G4GPU_CUDA_CHECK(cudaMemset(d_qcount_, 0, sizeof(int) * kNB));
      count_interactions<real_t><<<qblocks, threads_>>>(items, n, d_qcount_);
      int nb[kNB] = {};
      G4GPU_CUDA_CHECK(cudaMemcpy(nb, d_qcount_, sizeof(int) * kNB, cudaMemcpyDeviceToHost));
      InteractionOffsets qoff{};
      int acc = 0;
      for (int b = 0; b < kNB; ++b) { qoff.base[b] = acc; acc += nb[b]; }
      G4GPU_CUDA_CHECK(cudaMemset(d_qcount_, 0, sizeof(int) * kNB));
      scatter_interactions<real_t><<<qblocks, threads_>>>(items, n, d_qidx_, qoff, d_qcount_);
      prof.memop(2);
      prof.launch(2);
      prof.sync();
      prof.end(LoopProfile::kQueueBin, n);

      // A BLOCK OF ONE THREAD, since P32 (docs/RISK.md V285), and not the 32 it was. The
      // 32 was chosen so that a chunk of 40 interactions would use two SMs rather than one;
      // what it also did was put 32 different cascades in one WARP, and a cascade is the
      // most divergent code in the port - every thread of the warp at a different
      // collision, a different nucleus, a different branch - so the warp ran its 32
      // cascades nearly one after another, on 4 of the card's 80 SMs for a 128-slot chunk.
      // One thread a block gives each interaction a warp of its own, spreads a chunk over
      // every SM, and leaves the scheduler 32 resident warps an SM to hide each one's
      // local-memory latency behind. Measured on the 1 GeV proton beam (V283): 106.8 s at
      // 32, 29.9 s at 2, 26.8 s at 1, the dose to the digit each time. The idle lanes cost
      // nothing the divergent warp was not already wasting.
      //
      // `G4GPU_INTERACTION_BLOCK=n` overrides it for the experiment. Thread `i` of a chunk
      // still takes slot `i` and queue entry `base + i` whatever the block, so the block
      // moves which SM runs an interaction and nothing it computes.
      int kIth = 1;
      if (const char* env = std::getenv("G4GPU_INTERACTION_BLOCK")) {
        const int v = std::atoi(env);
        if (v > 0 && v <= 1024) { kIth = v; }
      }
      // `G4GPU_INTERACTION_CHUNKED=1`: the chunk-at-a-time schedule V295 replaced, for the
      // comparison.
      static const bool chunked = [] {
        const char* v = std::getenv("G4GPU_INTERACTION_CHUNKED");
        return v != nullptr && v[0] == '1';
      }();
#define G4GPU_DRAIN(BUCKET)                                                              \
  do {                                                                                       \
    const int b_ = static_cast<int>(BUCKET);                                                 \
    const int pc_ = LoopProfile::kIntBase + (b_ - static_cast<int>(had::InteractionBucket::kFtfp)); \
    if (nb[b_] > 0) { prof.begin(pc_); }                                                     \
    /* P32: the two buckets that reach FTFP are chunked at FTFP's own count (V285). */      \
    const int w_ = (BUCKET == had::InteractionBucket::kFtfp                                  \
                    || BUCKET == had::InteractionBucket::kAtRest)                            \
                       ? std::min(n_interaction_slots_, interaction_pool_.n_ftf_slots)       \
                       : n_interaction_slots_;                                               \
    /* P34 (V295): one launch, at most w_ threads, each walking the bucket with a stride;   \
       chunked (the old schedule) on request. */                                             \
    const int chunk_ = chunked ? w_ : nb[b_];                                                \
    for (int base = 0; base < nb[b_]; base += chunk_) {                                      \
      const int n_this = (nb[b_] - base < chunk_) ? (nb[b_] - base) : chunk_;                \
      const int width_ = (n_this < w_) ? n_this : w_;                                        \
      const int th_ = (kIth < width_) ? kIth : width_;                                       \
      const int blocks_ = chunked ? (n_this + kIth - 1) / kIth : width_ / th_;               \
      run_interaction<real_t, BUCKET><<<blocks_, chunked ? kIth : th_>>>(                    \
          scene_, items, d_qidx_, n_this, qoff.base[b_] + base, out, stride,                 \
          d_win_score, d_voxel_score_, wiring, interaction_pool_.d_pool, traj,               \
          d_status_warn_, sec_, bk, hook_);                                                  \
      ++interaction_chunks_;                                                                 \
      prof.launch();                                                                         \
    }                                                                                        \
    if (nb[b_] > 0) { prof.end(pc_, nb[b_]); }                                               \
  } while (0)
      G4GPU_DRAIN(had::InteractionBucket::kFtfp);
      G4GPU_DRAIN(had::InteractionBucket::kBertini);
      if constexpr (std::is_same<StepHook, StepTap<double>>::value &&
                    std::is_same<real_t, double>::value) {
        constexpr int bb_ = static_cast<int>(had::InteractionBucket::kBinary);
        if (flat_pool_().ok() && nb[bb_] > 0) {
          // P36: the flat path (docs/RISK.md V309). What it hands back runs on the thread path
          // in the same drain, from `strag`, before anything else steps.
          prof.begin(LoopProfile::kIntBase + (bb_ - static_cast<int>(had::InteractionBucket::kFtfp)));
          const int ns_ = flat_binary_drain(
              flat_pool_(), scene_, items, d_qidx_, nb[bb_], qoff.base[bb_], out, stride,
              d_win_score, d_voxel_score_, wiring, interaction_pool_.d_pool,
              n_interaction_slots_, traj, d_status_warn_, sec_, bk, hook_, flat_stats_());
          if (ns_ > 0) {
            const int w_ = (ns_ < n_interaction_slots_) ? ns_ : n_interaction_slots_;
            run_interaction<real_t, had::InteractionBucket::kBinary><<<w_, 1>>>(
                scene_, items, flat_pool_().strag, ns_, 0, out, stride, d_win_score,
                d_voxel_score_, wiring, interaction_pool_.d_pool, traj, d_status_warn_, sec_,
                bk, hook_);
            ++interaction_chunks_;
          }
          prof.end(LoopProfile::kIntBase + (bb_ - static_cast<int>(had::InteractionBucket::kFtfp)),
                   nb[bb_]);
        } else {
          G4GPU_DRAIN(had::InteractionBucket::kBinary);
        }
      } else {
        G4GPU_DRAIN(had::InteractionBucket::kBinary);
      }
      if constexpr (std::is_same<StepHook, StepTap<double>>::value &&
                    std::is_same<real_t, double>::value) {
        constexpr int bl_ = static_cast<int>(had::InteractionBucket::kLightIon);
        if (flat_ion_pool_().ok() && nb[bl_] > 0) {
          // P36 (docs/RISK.md V310): the light-ion reaction's flat path; its hand-backs run on
          // the thread path in the same drain.
          prof.begin(LoopProfile::kIntBase + (bl_ - static_cast<int>(had::InteractionBucket::kFtfp)));
          const int ns_ = flat_ion_drain(
              flat_ion_pool_(), scene_, items, d_qidx_, nb[bl_], qoff.base[bl_], out, stride,
              d_win_score, d_voxel_score_, wiring, interaction_pool_.d_pool,
              n_interaction_slots_, traj, d_status_warn_, sec_, bk, hook_, flat_ion_stats_());
          if (ns_ > 0) {
            const int w_ = (ns_ < n_interaction_slots_) ? ns_ : n_interaction_slots_;
            run_interaction<real_t, had::InteractionBucket::kLightIon><<<w_, 1>>>(
                scene_, items, flat_ion_pool_().strag, ns_, 0, out, stride, d_win_score,
                d_voxel_score_, wiring, interaction_pool_.d_pool, traj, d_status_warn_, sec_,
                bk, hook_);
            ++interaction_chunks_;
          }
          prof.end(LoopProfile::kIntBase + (bl_ - static_cast<int>(had::InteractionBucket::kFtfp)),
                   nb[bl_]);
        } else {
          G4GPU_DRAIN(had::InteractionBucket::kLightIon);
        }
      } else {
        G4GPU_DRAIN(had::InteractionBucket::kLightIon);
      }
      G4GPU_DRAIN(had::InteractionBucket::kAtRest);
#undef G4GPU_DRAIN
      // `kNone` is not drained and is not silent: the stepper booked `kNoInelasticModel` for
      // every entry in it before it enqueued, and an entry with no model has no kernel to run.
      // It is counted here so the report can say how many interactions never reached a launch.
      interactions_no_model_ += nb[static_cast<int>(had::InteractionBucket::kNone)];
      G4GPU_CUDA_CHECK(cudaGetLastError());
    };
    // The pending ring, drained into `out`, emptied.
    auto drain_pending = [&](TrackBuffer<real_t> out, const had::HadronicWiring<real_t>& wiring,
                             const EmitterBooks& bk) {
      if (pending_n <= 0) { return; }
      drain_interactions(d_pending, pending_n, out, wiring, bk);
      ++deferred_drains;
      pending_n = 0;
    };
    cudaEvent_t t0, t1;
    G4GPU_CUDA_CHECK(cudaEventCreate(&t0));
    G4GPU_CUDA_CHECK(cudaEventCreate(&t1));
    G4GPU_CUDA_CHECK(cudaEventRecord(t0));

    // Batches, with a retry if one runs out of buffer space.
    //
    // A track that cannot be appended is a track that stops being transported, and its energy
    // never reaches a scorer - so a run that drops one reports a dose that is too low while
    // looking exactly like a result. Ending the run on that is safe but wasteful: the work is
    // thrown away and the user is left to guess a setting.
    //
    // Retrying is better and costs nothing but the attempt. The arena is a fixed size, so
    // HALVING THE BATCH DOUBLES THE SLOTS PER EVENT - the same memory, fewer events sharing
    // it - which is exactly the shortage that caused the overflow. And the retry is not an
    // approximation of the run that failed: every track's random stream is keyed on its GLOBAL
    // event index, so events transported in a smaller batch draw precisely what they would
    // have drawn in a larger one. The answer is the one a correctly sized run would have given.
    //
    // Nothing is accumulated until an attempt completes without dropping anything, so a failed
    // attempt contributes nothing to the totals.
    int base = 0;
    int try_batch = batch_;
    while (base < n_events) {
      // P32: `n_batch` is the WINDOW's event count - the events this pass scores - and
      // `n_first` how many of them are seeded at its first iteration. Without a window wider
      // than the batch (one batch, refill off, or no memory for one) the two are equal and this
      // is the batch loop it was, a retry's halving included.
      const int n_batch = std::min((win_cap > batch_) ? win_cap : try_batch, n_events - base);
      const int n_first = std::min(try_batch, n_batch);
      G4GPU_CUDA_CHECK(cudaMemset(d_win_score, 0, sizeof(double) * n_scorers_ * stride));
      // Zeroed with the scores and for the same reason: it is indexed by the BATCH-local event
      // id, so a batch that reused a slot would add this batch's neutrinos to the last one's.
      if (d_win_carried != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemset(d_win_carried, 0, sizeof(double) * stride));
      }
      // This attempt's drops, not a previous attempt's.
      for (int i = 0; i < 2; ++i) { tracks_[i].reset_overflow(); }
      const long long steps_before = st.track_steps;
      const long long abandoned_before = st.abandoned;

      int cur = 0, nxt = 1;
      pending_n = 0;  // a retried window starts with nothing pending (P34)
      tracks_[cur].reset();
      G4GPU_CUDA_CHECK(cudaMemcpy(d_win_prim, primaries + base,
                                  sizeof(Primary<real_t>) * n_batch, cudaMemcpyHostToDevice));
      // No partition to do: the primaries all go into the one pool.
      //
      // This is where five buffers hurt most. Every primary needs a slot at the same instant,
      // and in a gamma beam they are all one species, so a fixed split gave that species its
      // fraction and dropped the rest - and the throttle could not help, because nothing had
      // been stepped yet. With one pool the only question is whether the batch fits, which is
      // one comparison rather than a division nobody could get right in advance.
      if (n_first > tracks_[cur].view.capacity) {
        std::printf("\nFATAL: %d primaries do not fit a pool of %d slots. Raise\n"
                    "       SetLiveTracksPerEvent (now %.1f per event) or lower the batch.\n",
                    n_first, tracks_[cur].view.capacity, live_per_event_);
        std::exit(3);
      }
      seed_from_primaries<real_t><<<(n_first + threads_ - 1) / threads_, threads_>>>(
          tracks_[cur].view, geom_, d_win_prim, n_first, key_base_for(stream_pos, base),
          seed);
      G4GPU_CUDA_CHECK(cudaGetLastError());
      int next = n_first;    // the window's next unseeded event
      int last_seed = 0;     // the iteration the youngest track's event was seeded in

      // Bounded so a non-terminating track shows up as an explicit abandonment rather than a
      // hang - RunAction prints the count. B1's photon beam drains in about 110 iterations, but
      // an iteration is one step of every live track and a real nuclide takes thousands: a
      // 1 GeV per nucleon Fe56 takes about 3,400 steps to stop in B1's phantom (Geant4 takes
      // 3,593 real-ion steps an event on that beam) and its batch drained at 10,030 iterations,
      // so the old bound of 1,000 abandoned every Fe56 before the bone and the row read 3.88 pGy
      // against Geant4's 12.6 uGy (P30, docs/RISK.md V276-V277). 20,000 is twice the deepest
      // drain measured; a beam that needs more is still reported, never hidden.
      //
      // P32: THE BOUND IS PER TRACK NOW, NOT PER BATCH. With the refill a window's iterations
      // are not one generation of tracks: an event seeded at iteration 900 has had 900 fewer
      // than the window's first. So the drain stops at `kMaxIterations` past the iteration the
      // YOUNGEST event was seeded in (`last_seed`, which a repair pass also resets) - every track
      // still alive then has had at least that many iterations, which is the guarantee the
      // batch's bound gave its own tracks - and what it abandons is counted as before. A window
      // seeded whole has `last_seed` 0 and the bound it always had.
      constexpr int kMaxIterations = 20000;
      int leftover = 0;
      for (int iteration = 0; iteration - last_seed < kMaxIterations; ++iteration) {
        // How many tracks are alive, of any species. One number, because there is one pool.
        prof.start_iteration();
        int n_live = live_count(tracks_[cur]);  // P37 (V304): a mapped read-back
        prof.sync();  // TrackBuffer::count is a device-to-host copy of the append cursor

        // ---- P32: the refill. The window's next primaries go into the room the pool has
        // freed, as many as bring the live count back to the batch - never more than the first
        // iteration seeded, so the throttle below has at least the room it had then. They are
        // appended after the live tracks (`count` is the append cursor, and nothing else is
        // appended between here and the counting sort), so the live count afterwards is
        // `n_live + k` without another read-back.
        if (next < n_batch && n_live < try_batch) {
          const int k = std::min(n_batch - next, try_batch - n_live);
          seed_from_primaries<real_t><<<(k + threads_ - 1) / threads_, threads_>>>(
              tracks_[cur].view, geom_, d_win_prim, k, key_base_for(stream_pos, base), seed,
              nullptr, next);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          prof.launch();
          next += k;
          n_live += k;
          last_seed = iteration;
          ++refills;
          refilled_events += k;
        }

        // ---- P34: nothing else alive and interactions still pending (V294): they are drained
        // into this iteration's pool before it is sorted, so the window cannot end on them. And
        // when their reservation would take half the room the throttle has, for the same place:
        // the pool's room goes to stepping, not to holding interactions back.
        if (pending_n > 0
            && (n_live == 0
                || static_cast<long long>(pending_n) * max_secondaries_per_step(kSpeciesProton)
                       > (static_cast<long long>(tracks_[nxt].view.capacity) - n_live) / 2)) {
          drain_pending(tracks_[cur].view, last_wiring, last_books);
          n_live = live_count(tracks_[cur]);
          prof.sync();
        }

        // The dispatch layout for this iteration: which slots hold gammas, which electrons,
        // and so on. Rebuilt every time because the pool is written in append order, which is
        // whatever the atomics gave out, and a kernel needs its own species contiguous.
        //
        // Two passes and a prefix sum - a counting sort into the one index array. The counts
        // have to come back to the host anyway to size the launches, so the sum costs nothing
        // extra beyond the second launch.
        int nsp[kNumTrackSpecies] = {};
        SpeciesOffsets off{};
        if (n_live > 0) {
          const int blocks = (n_live + threads_ - 1) / threads_;
          // P33 (V291): the buckets are (species, material, energy); `G4GPU_SORT_KEYS=0` puts
          // every track of a species in its first bucket, which is the species-only sort.
          static const int sort_keyed = [] {
            const char* v = std::getenv("G4GPU_SORT_KEYS");
            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 1;
          }();
          constexpr int kBuckets = kNumTrackSpecies * kSortKeys;
          static thread_local std::vector<int> bucket(kBuckets);
          prof.begin(LoopProfile::kSort);
          // P37 (V304): the prefix sum on the card, and the 40 integers the launches need written
          // by it into mapped host memory - no copy, one synchronisation. `G4GPU_SORT_DEVICE=0` is
          // the host's sum below, as before. (A one-launch sort for small pools, count, sum and
          // scatter in one block, was measured as no faster - 40.44 against 40.55 s - and is not
          // kept: what the sort costs on the ion beams is the synchronisation, not its launches.)
          static const bool sort_device = [] {
            const char* v = std::getenv("G4GPU_SORT_DEVICE");
            return !(v != nullptr && v[0] == '0' && v[1] == '\0');
          }();
          const bool on_device = sort_device && d_sort_summary != nullptr;
          G4GPU_CUDA_CHECK(cudaMemset(idx_n_[cur], 0, sizeof(int) * kBuckets));
          count_species<real_t><<<blocks, threads_>>>(tracks_[cur].view, n_live,
                                                      scene_.geometry.volumes, sort_keyed,
                                                      idx_n_[cur], d_unknown_);
          G4GPU_CUDA_CHECK(cudaGetLastError());
          if (on_device) {
            scan_species_buckets<<<1, kSortScanThreads>>>(idx_n_[cur], d_sort_summary, d_map);
            G4GPU_CUDA_CHECK(cudaGetLastError());
            scatter_species<real_t><<<blocks, threads_>>>(tracks_[cur].view, n_live,
                                                          scene_.geometry.volumes, sort_keyed,
                                                          idx_[cur], idx_n_[cur]);
            G4GPU_CUDA_CHECK(cudaGetLastError());
            G4GPU_CUDA_CHECK(cudaStreamSynchronize(0));
            const volatile int* hs = h_sort_summary;
            for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
              off.base[sp] = hs[sp];
              nsp[sp] = hs[kNumTrackSpecies + sp];
            }
            prof.memop();
            prof.launch(3);
            prof.sync();
            prof.end(LoopProfile::kSort, n_live);
          } else {
            G4GPU_CUDA_CHECK(cudaMemcpy(bucket.data(), idx_n_[cur], sizeof(int) * kBuckets,
                                        cudaMemcpyDeviceToHost));
            prof.memop(2);
            prof.launch(2);
            prof.sync();
            // Each bucket's count becomes its starting cursor, in species-major order, so a
            // species' buckets stay one contiguous range starting at `off.base[sp]`.
            int at = 0;
            for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
              off.base[sp] = at;
              for (int k = 0; k < kSortKeys; ++k) {
                const int c = bucket[sp * kSortKeys + k];
                bucket[sp * kSortKeys + k] = at;
                at += c;
              }
              nsp[sp] = at - off.base[sp];
            }
            // Overwritten with the cursors and reused by the scatter; see scatter_species.
            G4GPU_CUDA_CHECK(cudaMemcpy(idx_n_[cur], bucket.data(), sizeof(int) * kBuckets,
                                        cudaMemcpyHostToDevice));
            prof.memop();
            scatter_species<real_t><<<blocks, threads_>>>(tracks_[cur].view, n_live,
                                                          scene_.geometry.volumes, sort_keyed,
                                                          idx_[cur], idx_n_[cur]);
            G4GPU_CUDA_CHECK(cudaGetLastError());
            prof.end(LoopProfile::kSort, n_live);
          }
        }
        if (n_live == 0) {
          st.max_iterations = std::max(st.max_iterations, iteration);

          // Every track is done - unless some were dropped, in which case the events that lost
          // them are transported again, here, and nothing else is.
          //
          // This is the whole reason the drop list exists. Discarding the batch would redo a
          // million healthy events to repair one shower; redoing the events that actually lost
          // something costs their showers and nothing more. The answer is the same either way,
          // because a track's random stream is keyed on its global event index and not on when
          // or alongside what it was transported.
          int n_dropped = 0;
          if (d_dropped_count_ != nullptr) {
            G4GPU_CUDA_CHECK(cudaMemcpy(&n_dropped, d_dropped_count_, sizeof(int),
                                        cudaMemcpyDeviceToHost));
          }
          if (n_dropped == 0) { break; }
          if (n_dropped > dropped_capacity_) {
            // More events were hit than the list can name, so it is a partial record and there
            // is no way to know which events to repair. That is not an outlier - it is a pool
            // that is too small for this problem - and the batch retry outside handles it.
            break;
          }

          std::vector<int> ids(static_cast<size_t>(n_dropped));
          G4GPU_CUDA_CHECK(cudaMemcpy(ids.data(), d_dropped_, sizeof(int) * n_dropped,
                                      cudaMemcpyDeviceToHost));
          std::sort(ids.begin(), ids.end());
          ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
          // An id outside the batch would mean the list itself is corrupt; drop the pass rather
          // than seed from it.
          ids.erase(std::remove_if(ids.begin(), ids.end(),
                                   [&](int e) { return e < 0 || e >= n_batch; }),
                    ids.end());
          if (ids.empty()) { break; }

          // What those events deposited on the attempt that lost a track is not a partial
          // answer to be added to - it is wrong, and it is replaced. Zeroing first is what
          // makes the repair a redo rather than a double count.
          for (int e : ids) {
            for (int sc = 0; sc < n_scorers_; ++sc) {
              G4GPU_CUDA_CHECK(cudaMemset(d_win_score + static_cast<size_t>(sc) * stride + e, 0,
                                          sizeof(double)));
            }
            // The carried-away ledger is per event exactly as the score is, so a repaired
            // event's neutrinos have to be forgotten alongside its deposit. Leaving it would
            // double-count every neutrino of every repaired event - an error that shows up
            // only as an energy balance that does not close, which is the last place anyone
            // looks.
            if (d_win_carried != nullptr) {
              G4GPU_CUDA_CHECK(cudaMemset(d_win_carried + e, 0, sizeof(double)));
            }
          }
          st.events_repaired += static_cast<long long>(ids.size());
          std::printf("repair: %zu of %d events lost a track and are being transported\n"
                      "        again; the rest of the batch is untouched\n",
                      ids.size(), n_batch);

          const int n_repair = static_cast<int>(ids.size());
          G4GPU_CUDA_CHECK(cudaMemcpy(d_dropped_, ids.data(), sizeof(int) * n_repair,
                                      cudaMemcpyHostToDevice));
          G4GPU_CUDA_CHECK(cudaMemset(d_dropped_count_, 0, sizeof(int)));

          tracks_[cur].reset();
          seed_from_primaries<real_t><<<(n_repair + threads_ - 1) / threads_, threads_>>>(
              tracks_[cur].view, geom_, d_win_prim, n_repair,
              key_base_for(stream_pos, base), seed, d_dropped_);
          last_seed = iteration;  // the repaired events start their tracks afresh
          G4GPU_CUDA_CHECK(cudaGetLastError());
          continue;  // drain again, this time carrying only the repaired events
        }
        if (iteration - last_seed == kMaxIterations - 1) { leftover = n_live; }

        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          st.peak_live[sp] = std::max(st.peak_live[sp], nsp[sp]);
        }

        // How many tracks of each species may be stepped this iteration.
        //
        // Nothing is re-divided any more. There is one output pool, so there is one budget:
        // every live track needs a slot in it whether it is stepped or merely passed through,
        // and what is left over after that is what pays for secondaries.
        //
        // The fan-out reservation is PER SPECIES - max_secondaries_per_step in
        // core/track_buffer.cuh - and the budget below is therefore counted in SLOTS rather
        // than in tracks. It was one constant of four for every species, which was correct
        // while every species could make at most three secondaries in a step. A hadronic
        // inelastic reaction makes tens, and raising a shared constant to cover it would
        // divide the gamma's budget by the same factor: on B1's own numbers, ~2 spare slots an
        // event at a reservation of 4 is 0.5 tracks an event, and at 64 it is 0.03 - sixty
        // times the iterations to drain a gamma run that will never make a neutron. Every
        // reservation is still 4 today, so this loop chooses exactly what the old one did.
        //
        // A species that cannot be stepped in full this iteration is not truncated: the
        // remainder passes through untouched and is stepped next time.
        const int cap_out = tracks_[nxt].view.capacity;
        int slots = (cap_out > n_live) ? (cap_out - n_live) : 0;
        // P34 (V294): a pending interaction keeps the reservation its own step made, until it
        // is drained into some iteration's output pool.
        if (pending_n > 0) {
          const long long held =
              static_cast<long long>(pending_n) * max_secondaries_per_step(kSpeciesProton);
          slots = (slots > held) ? static_cast<int>(slots - held) : 0;
        }
        int step_n[kNumTrackSpecies] = {};
        int total_step = 0;
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          const int per = max_secondaries_per_step(sp);
          const int can = (per > 0) ? (slots / per) : nsp[sp];
          step_n[sp] = std::min(nsp[sp], can);
          slots -= step_n[sp] * per;
          total_step += step_n[sp];
        }
        if (total_step == 0) {
          std::printf("\nFATAL: no track can be stepped without overrunning the pool.\n"
                      "       %d tracks are live and the pool holds %d slots a side. Raise\n"
                      "       SetLiveTracksPerEvent (now %.1f per event) or lower the batch.\n",
                      n_live, cap_out, live_per_event_);
          std::exit(3);
        }
        st.throttled += n_live - total_step;
        st.track_steps += total_step;

        tracks_[nxt].reset();
        prof.memop();
        // The secondary arena holds one iteration worth of chains and no more: the chains
        // a stepping action walks are the ones made by the launch it is running in, and
        // nothing reads them afterwards. Resetting here is what keeps the arena sized
        // against tracks in flight rather than against the length of the run.
        if (sec_.cursor != nullptr) {
          G4GPU_CUDA_CHECK(cudaMemsetAsync(sec_.cursor, 0, sizeof(int)));
          prof.memop();
        }
        // ---- can anything be queued this iteration at all?
        //
        // A TRACK THAT IS NOT A HADRON CANNOT ENQUEUE, so an iteration with no hadronic species
        // live needs no cursor rewind, no read-back and no drain - and the read-back is the
        // expensive half, because `cudaMemcpy` D2H is a full device synchronisation and this
        // loop only had one of those (the species histogram) before P15.
        //
        // MEASURED, AND IT IS WHY THIS TEST IS HERE: B1's 2,000,000-event 6 MeV gamma gate went
        // from 1,172 ms of event loop to 1,947 ms with the drain running unconditionally - 1.7
        // times slower on a run that cannot produce one interaction. It is not memory: halving
        // the pool with `G4GPU_LIVE_PER_EVENT=2` moved it by 1.4%, which rules out the 368 MB
        // the queue was taking. It is the second synchronisation per iteration, 88 of them in
        // that run.
        //
        // `nsp[]` is the species histogram the counting sort already read back, so this costs
        // nothing: it is a sum over the hadronic rows of a `kNumTrackSpecies` array on the host.
        //
        // NOT EVERY HADRON CAN QUEUE, and since P19 the difference is on the gamma gate's clock:
        // a photo-nuclear reaction makes a neutron, and in `kStage1` - every gate's stage - a
        // neutron has elastic scattering and capture and NO process that queues (`step_neutral`'s
        // `has_stage1_elastic` / `has_stage1_capture` have no inelastic sibling), and a pi0
        // decays in every stage. So neither holds this read-back open. Everything else that is
        // a hadron here can queue something - an inelastic interaction or an at-rest capture.
        const bool neutron_can_queue = [&] {
          had::HadronicStage stage = had_stage_;
          if (const char* env = std::getenv("G4GPU_HADRONIC_STAGE")) {
            if (env[0] == 'f') { stage = had::HadronicStage::kFinal; }
            else if (env[0] == 's') { stage = had::HadronicStage::kStage1; }
          }
          return stage == had::HadronicStage::kFinal;
        }();
        bool any_hadron = false;
        for (int sp = 0; sp < kNumTrackSpecies && !any_hadron; ++sp) {
          if (nsp[sp] <= 0 || max_secondaries_per_step(sp) <= 4) { continue; }
          if (sp == kSpeciesPiZero) { continue; }
          if (sp == kSpeciesNeutron && !neutron_can_queue) { continue; }
          any_hadron = true;
        }
        // THE STACK IS NOT RAISED HERE ANY MORE, but where an interaction kernel is about to
        // launch (the drain below, once the read-back says something was queued). P19 is why:
        // a photon run makes hadrons now - B1's 6 MeV gamma gate has about one photo-nuclear
        // reaction in 2,000,000 events, and its neutron or recoil nucleus made this line
        // reserve 94,208 bytes a thread for kernels that never launched. Measured on that run:
        // the raise left 0.00 GB of the card, WDDM paged, and the event loop went from about
        // 1.2 s to 13.5 s (docs/RISK.md V210). Every launch of the five kernels is inside
        // `n_queued > 0`, so moving the call there changes nothing for a run that interacts.
        ++iterations_run_;
        if (any_hadron) { ++hadron_iterations_; }
        if (nsp[kSpeciesNeutron] > 0) { ++neutron_iterations_; }
        if (any_hadron && d_queue_cursor_ != nullptr) {
          // The interaction queue holds ONE iteration's interactions and is drained at the end
          // of it, so its cursor is rewound here beside the arena's and for the same reason:
          // the bound that says it cannot overflow is "one launch's tracks", not "one run's".
          G4GPU_CUDA_CHECK(cudaMemsetAsync(d_queue_cursor_, 0, sizeof(int)));
          prof.memop();
        }
        // P19's queue, rewound on EVERY iteration and asynchronously - it is drained on the
        // device at the end of the same iteration, so no host read ever waits on it, and an
        // iteration that queues nothing costs this memset and two empty launches.
        const bool any_emx = (nsp[kSpeciesGamma] > 0 || nsp[kSpeciesElectron] > 0
                              || nsp[kSpeciesPositron] > 0 || nsp[kSpeciesMuonMinus] > 0
                              || nsp[kSpeciesMuonPlus] > 0);
        if (any_emx && d_emx_cursor_ != nullptr) {
          G4GPU_CUDA_CHECK(cudaMemsetAsync(d_emx_cursor_, 0, sizeof(int)));
          prof.memop();
        }
        // One launch per species, each over its own range of the one index list, each into the
        // one output pool.
        //
        // A specialised kernel per species remains because a warp whose threads take different
        // physics paths serialises through all of them - that is the reason the species list
        // survives at all. What has gone is the idea that a specialised KERNEL needs a separate
        // BUFFER, and now also the idea that it needs a separate index ARRAY.
        //
        // The switch is written out rather than driven from a table of function pointers
        // because a __global__ template's address is not something a host table can hold
        // portably, and because each line names the specialisation it launches - which is what
        // makes a missing species a compile error in the switch rather than a track that is
        // counted and never stepped.
        EmitterBooks books{d_win_carried, d_carried_n_, d_refused_, d_refused_e_};
        // P23a's study knob: `G4GPU_STRANGE_TRACKS=0` refuses the sixteen strange species at
        // emission as the port did before they had kernels - see `EmitterBooks::refuse_strange`.
        if (const char* env = std::getenv("G4GPU_STRANGE_TRACKS")) {
          books.refuse_strange = (env[0] == '0');
        }
        // P8's wiring, built here and passed by value into every hadronic launch. One struct
        // rather than five arguments, and built per launch rather than at Upload so that the
        // stage can change between two BeamOn calls in one process - which is what
        // ref/b1hadron/stage1_compare.ps1 needs of it.
        had::HadronicWiring<real_t> had_wiring{};
        // `G4GPU_HADRONIC_STAGE=final` (or `stage1`) overrides the engine's default, for the
        // same reason `G4GPU_LIVE_PER_EVENT` and `G4GPU_ION_INELASTIC` exist: `examples/B1` is
        // Geant4's own B1 and has no UI command that reaches a setter.
        //
        // AND UNTIL P15 THERE WAS NOTHING IN THIS REPOSITORY THAT SELECTED `kFinal` AT ALL.
        // The default is `kStage1`, which for a neutron means `hadElastic` and `nCapture` as two
        // separate processes and NO inelastic - see `step_neutral`'s `has_stage1_elastic` /
        // `has_stage1_capture`, which have no inelastic sibling. So P15's neutron inelastic
        // sub-process, the port's largest named hole from P8d, was wired and then reachable only
        // from `tests/test_inelastic_transport.cu`, which sets the stage itself. A deliverable
        // no production run can reach is not a deliverable. docs/RISK.md V194.
        //
        // The DEFAULT is deliberately left at `kStage1` here rather than flipped: every gate in
        // this project was measured in it, and which stage ships is a decision about all of them
        // and not about P15. What P15 owes is that the final stage can be selected and has been
        // measured, which is what this variable and the sweep's use of it are.
        had::HadronicStage stage = had_stage_;
        if (const char* env = std::getenv("G4GPU_HADRONIC_STAGE")) {
          if (env[0] == 'f') { stage = had::HadronicStage::kFinal; }
          else if (env[0] == 's') { stage = had::HadronicStage::kStage1; }
        }
        had_wiring.stage = stage;
        // What the report prints: the stage this run USED, which `G4GPU_HADRONIC_STAGE` may have
        // changed from `had_stage_`. The ledger header printed `had_stage_` until V200 and said
        // "stage1" over runs whose neutrons were visibly reacting inelastically.
        had_stage_used_ = stage;
        // P22: the neutron general process's order, overridden per launch by
        // `G4GPU_NEUTRON_GP_ORDER` for the stage variable's reason. `reordered` (or anything
        // starting with `r`) is the order every neutron row before P22 was measured in; `11.1.1`
        // (or `g4`/`geant4`) is the default. Anything else is NOT ignored quietly - a misspelled
        // override that left the default on would be a run labelled with the wrong physics -
        // so it is reported, and the report line names what ran.
        had::NeutronGeneralOrder neutron_order = neutron_order_;
        if (const char* env = std::getenv("G4GPU_NEUTRON_GP_ORDER")) {
          if (env[0] == 'r' || env[0] == 'R') {
            neutron_order = had::NeutronGeneralOrder::kReordered;
          } else if (env[0] == '1' || env[0] == 'g' || env[0] == 'G') {
            neutron_order = had::NeutronGeneralOrder::kGeant4_11_1_1;
          } else if (env[0] != '\0' && !neutron_order_env_warned_) {
            std::printf("WARNING: G4GPU_NEUTRON_GP_ORDER=\"%s\" is neither 11.1.1 nor reordered; "
                        "the engine's setting (%s) is used\n",
                        env, had::neutron_general_order_key(neutron_order_));
            neutron_order_env_warned_ = true;
          }
        }
        had_wiring.neutron_order = neutron_order;
        neutron_order_used_ = neutron_order;
        had_wiring.decay = had_decay_;
        had_wiring.hadron_elastic = had_elastic_;
        had_wiring.neutron_capture = had_capture_;
        // Five device pointers, copied by value into the launch like the rest of the struct.
        // Null in a run whose scene can see no hadron - Upload only builds them when it builds
        // the hadron range table - and `elastic_xs_per_volume` then returns zero, which is the
        // same "no process" state a lepton is in.
        had_wiring.elastic = elastic_tables_.view;
        // The neutron's two per-process cross sections. Read in `kStage1` as two whole processes
        // with their own interaction lengths, and in `kFinal` as the data stores the chosen
        // sub-process draws its target from. `host/neutron_upload.cuh`.
        had_wiring.neutron = neutron_tables_.sub;
        // P3's level scheme. Reached since P8d - the capture sub-process walks it - and
        // `Upload`'s refusal above is what keeps it from arriving separately from the cross
        // sections.
        had_wiring.level_data = level_tables_.view;
        // P15's two processes and the queue they go through. The tables are null in a run
        // whose G4PARTICLEXSDATA could not be resolved, which is the same "no process" state a
        // species with no inelastic channel is in.
        had_wiring.hadron_inelastic = had_inelastic_;
        // `G4GPU_ION_INELASTIC=0` switches the five ion names off without rebuilding B1, the
        // way `G4GPU_LIVE_PER_EVENT` sets the live-track pool. Read here rather than cached at
        // construction, so a sweep that sets it per beam gets it per beam.
        bool ion_inel = had_ion_inelastic_;
        if (const char* env = std::getenv("G4GPU_ION_INELASTIC")) {
          ion_inel = !(env[0] == '0' && env[1] == '\0');
        }
        had_wiring.ion_inelastic = ion_inel;
        had_wiring.hadron_at_rest = had_at_rest_;
        had_wiring.inelastic = inelastic_tables_.view;
        had_wiring.queue.items = d_queue_;
        had_wiring.queue.cursor = d_queue_cursor_;
        had_wiring.queue.capacity = queue_capacity_;
        had_wiring.books.count = d_had_refused_n_;
        had_wiring.books.energy = d_had_refused_e_;
        // P19: G4EmExtraPhysics' four processes. `G4GPU_PHOTON_NUCLEAR=0`,
        // `G4GPU_ELECTRO_NUCLEAR=0`, `G4GPU_MUON_NUCLEAR=0` hold each off without rebuilding B1,
        // the way `G4GPU_ION_INELASTIC` does; `G4GPU_GAMMA_GENERAL_PROCESS=0` is
        // `/process/em/UseGeneralProcess false`. Read here, per launch, for the same reason.
        auto env_on = [](const char* name, bool dflt) {
          const char* v = std::getenv(name);
          if (v == nullptr || v[0] == '\0') { return dflt; }
          return !(v[0] == '0' && v[1] == '\0');
        };
        had_wiring.photon_nuclear = env_on("G4GPU_PHOTON_NUCLEAR", had_photon_nuclear_);
        had_wiring.electro_nuclear = env_on("G4GPU_ELECTRO_NUCLEAR", had_electro_nuclear_);
        had_wiring.muon_nuclear = env_on("G4GPU_MUON_NUCLEAR", had_muon_nuclear_);
        had_wiring.gamma_general =
            env_on("G4GPU_GAMMA_GENERAL_PROCESS", gamma_general_on_) ? had::GammaGeneralProcess::kOn
                                                                     : had::GammaGeneralProcess::kOff;
        emx_config_used_[0] = had_wiring.photon_nuclear;
        emx_config_used_[1] = had_wiring.electro_nuclear;
        emx_config_used_[2] = had_wiring.muon_nuclear;
        emx_config_used_[3] = (had_wiring.gamma_general == had::GammaGeneralProcess::kOn);
        had_wiring.emextra = emextra_tables_.view;
        had_wiring.emx_queue.items = d_emx_queue_;
        had_wiring.emx_queue.cursor = d_emx_cursor_;
        had_wiring.emx_queue.capacity = emx_queue_capacity_;
        had_wiring.emx_stats = d_emx_stats_;
        last_wiring = had_wiring;
        last_books = books;

        // ---- P37: THE SPECIES' STEPPING LAUNCHES RUN AT ONCE, each on a stream of its own.
        //
        // A stepping launch on the ion beams is a few hundred to a few thousand tracks - 16
        // blocks of 128 for the Fe56 beam's GenericIon at 5,000 events, one or two for its
        // muons - and it lasts as long as ONE thread takes to step ONE track: 270 us for an ion,
        // 430 us for a proton (docs/RISK.md V303's per-phase timers), because nothing else is on
        // the card to hide that thread's latency. One stream ran the fifteen of an iteration one
        // after another, 1.39 ms an iteration on the 5k beam for 0.4 ms of the slowest. They
        // share nothing a launch order could decide: each reads its own index range, appends to
        // the one pool and adds to the one score through atomics, as the threads of one launch
        // already do, and every number a step draws is keyed on its track (V284) - so the order
        // the pool is written in, which V291's sort already changed, reaches no random number.
        //
        // AND THIS CARD DOES RUN THEM AT ONCE, which V294 seemed to say it does not. V294's
        // probe queued a long kernel on one stream and then, later, a short one on another,
        // which waited for everything queued before it: WDDM submits in order. Launched together
        // - forked from the legacy stream by an event, each recording its own and the legacy
        // stream waiting for all of them - ten 200 us kernels take 0.27 ms, not 2.0 (V304's
        // probe). The legacy stream is where every other launch of the loop is, so the join is
        // what keeps the drain and the swap after the last step, as the one stream did.
        //
        // `G4GPU_STEP_STREAMS=0` launches them on the legacy stream one after another, as before.
        static const bool step_streams = [] {
          const char* v = std::getenv("G4GPU_STEP_STREAMS");
          return !(v != nullptr && v[0] == '0' && v[1] == '\0');
        }();
        static cudaStream_t sp_stream[kNumTrackSpecies] = {};
        static cudaEvent_t sp_done[kNumTrackSpecies] = {};
        static cudaEvent_t sp_fork = nullptr;
        int n_launching = 0;
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) { n_launching += (nsp[sp] > 0) ? 1 : 0; }
        const bool forked = step_streams && n_launching > 1;
        if (forked) {
          if (sp_fork == nullptr) {
            G4GPU_CUDA_CHECK(cudaEventCreateWithFlags(&sp_fork, cudaEventDisableTiming));
            for (int k = 0; k < kNumTrackSpecies; ++k) {
              G4GPU_CUDA_CHECK(cudaStreamCreateWithFlags(&sp_stream[k], cudaStreamNonBlocking));
              G4GPU_CUDA_CHECK(cudaEventCreateWithFlags(&sp_done[k], cudaEventDisableTiming));
            }
          }
          prof.begin(LoopProfile::kStepStreams);
          G4GPU_CUDA_CHECK(cudaEventRecord(sp_fork, 0));
        }
        for (int sp = 0; sp < kNumTrackSpecies; ++sp) {
          const int n_sp = nsp[sp];
          if (n_sp <= 0) { continue; }
          cudaStream_t sst = 0;
          if (forked) {
            sst = sp_stream[sp];
            G4GPU_CUDA_CHECK(cudaStreamWaitEvent(sst, sp_fork, 0));
          }
          const int* list = idx_[cur] + off.base[sp];
          // P37 STUDY KNOB (V303: no effect): `G4GPU_STEP_BLOCK=n` launches in blocks of n
          // threads, `=0` spreads each launch over every SM (blocks of the smallest multiple of
          // 32 that does it, at most `threads_`). Unset, the launch is the one it always was.
          static const int step_block_env = [] {
            const char* v = std::getenv("G4GPU_STEP_BLOCK");
            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : -1;
          }();
          static const int n_sm = [] {
            int dev = 0, n = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
            return (n > 0) ? n : 1;
          }();
          int tpb = threads_;
          if (step_block_env > 0) {
            tpb = step_block_env;
          } else if (step_block_env == 0) {
            const int per_sm = (n_sp + n_sm - 1) / n_sm;
            tpb = std::min(threads_, std::max(32, ((per_sm + 31) / 32) * 32));
          }
          const int blocks = (n_sp + tpb - 1) / tpb;
          if (!forked) { prof.begin(LoopProfile::kStepBase + sp); }
#define G4GPU_LAUNCH_HADRON(TYPE)                                                            \
  run_step_hadron<real_t, TYPE><<<blocks, tpb, 0, sst>>>(                                \
      scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,             \
      d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books, hook_)
#define G4GPU_LAUNCH_NEUTRAL(TYPE)                                                           \
  run_step_neutral<real_t, TYPE><<<blocks, tpb, 0, sst>>>(                               \
      scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,             \
      d_voxel_score_, step_n[sp], d_neutron_xs_, had_wiring, d_killed_energy_, d_killed_n_,   \
      traj, d_status_warn_, sec_, books, hook_)
          switch (sp) {
            // The photon and the two leptons take `had_wiring` since P19, for `photonNuclear`,
            // `electronNuclear` and `positronNuclear`.
            case kSpeciesGamma:
              run_step_gamma<real_t><<<blocks, tpb, 0, sst>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,
                  d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books,
                  hook_);
              break;
            case kSpeciesElectron:
              run_step_lepton<real_t, false><<<blocks, tpb, 0, sst>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,
                  d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books,
                  hook_);
              break;
            case kSpeciesPositron:
              run_step_lepton<real_t, true><<<blocks, tpb, 0, sst>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,
                  d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books,
                  hook_);
              break;
            case kSpeciesProton:     G4GPU_LAUNCH_HADRON(ParticleType::kProton); break;
            case kSpeciesAlpha:      G4GPU_LAUNCH_HADRON(ParticleType::kAlpha); break;
            case kSpeciesMuonMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kMuonMinus); break;
            case kSpeciesMuonPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kMuonPlus); break;
            case kSpeciesPionPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kPionPlus); break;
            case kSpeciesPionMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kPionMinus); break;
            case kSpeciesKaonPlus:   G4GPU_LAUNCH_HADRON(ParticleType::kKaonPlus); break;
            case kSpeciesKaonMinus:  G4GPU_LAUNCH_HADRON(ParticleType::kKaonMinus); break;
            case kSpeciesAntiProton: G4GPU_LAUNCH_HADRON(ParticleType::kAntiProton); break;
            case kSpeciesDeuteron:   G4GPU_LAUNCH_HADRON(ParticleType::kDeuteron); break;
            case kSpeciesTriton:     G4GPU_LAUNCH_HADRON(ParticleType::kTriton); break;
            case kSpeciesHe3:        G4GPU_LAUNCH_HADRON(ParticleType::kHe3); break;
            // The sixteenth instantiation, and the only one whose tracks are not all the same
            // particle: each carries its own nuclide in `TrackState::ion_za`.
            case kSpeciesGenericIon: G4GPU_LAUNCH_HADRON(ParticleType::kGenericIon); break;
            case kSpeciesNeutron:    G4GPU_LAUNCH_NEUTRAL(ParticleType::kNeutron); break;
            case kSpeciesPiZero:     G4GPU_LAUNCH_NEUTRAL(ParticleType::kPiZero); break;
            // P23a's two GROUP kernels: each steps eight species and reads the species off the
            // track, so the launch names a group and not a ParticleType.
            case kSpeciesStrangeCharged:
              run_step_strange<real_t, true><<<blocks, tpb, 0, sst>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,
                  d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books,
                  hook_);
              break;
            case kSpeciesStrangeNeutral:
              run_step_strange<real_t, false><<<blocks, tpb, 0, sst>>>(
                  scene_, tracks_[cur].view, list, tracks_[nxt].view, n_sp, stride, d_win_score,
                  d_voxel_score_, step_n[sp], had_wiring, traj, d_status_warn_, sec_, books,
                  hook_);
              break;
            default:
              // Not reachable, and not silent if it becomes so: a TrackSpeciesIndex added
              // without a line above would land here, and a species counted by the histogram
              // and never launched is a track that stops being transported.
              std::printf("\nFATAL: species index %d has no stepping kernel, but %d tracks\n"
                          "       were sorted into its range. Add a case to the switch in\n"
                          "       TransportEngine::BeamOn.\n", sp, n_sp);
              std::exit(3);
          }
          if (forked) {
            G4GPU_CUDA_CHECK(cudaEventRecord(sp_done[sp], sst));
            G4GPU_CUDA_CHECK(cudaStreamWaitEvent(0, sp_done[sp], 0));
            prof.count_only(LoopProfile::kStepBase + sp, step_n[sp]);
          } else {
            prof.end(LoopProfile::kStepBase + sp, step_n[sp]);
          }
          prof.launch();
#undef G4GPU_LAUNCH_HADRON
#undef G4GPU_LAUNCH_NEUTRAL
        }
        if (forked) { prof.end(LoopProfile::kStepStreams, total_step); }
        G4GPU_CUDA_CHECK(cudaGetLastError());

        // ---- P15: drain the interaction queue, in chunks of `n_interaction_slots_`.
        //
        // THE CHUNKING IS THE WHOLE DESIGN, and it is what makes the slot count a capacity
        // whose shortage costs TIME rather than one whose shortage costs INTERACTIONS
        // (docs/RISK.md V188). Thread `i` of a chunk takes slot `i`, so `pool.slot(i)` is never
        // null and `kInteractionNoSlot` is a tripwire rather than a rate.
        //
        // AFTER every stepping launch and BEFORE the buffers swap, because the secondaries this
        // makes belong in `tracks_[nxt]` with the ones the steppers made: a cascade proton must
        // be stepped in the NEXT iteration, not this one. The read-back of the cursor is a
        // synchronising copy, which is the one place per iteration this loop waits for the
        // device - the counting sort above already reads `idx_n_` back for the same reason.
        if (any_hadron && d_queue_cursor_ != nullptr && interaction_pool_.ok) {
          int n_queued = 0;
          n_queued = read_back(d_queue_cursor_, kMapQueued);  // P37 (V304)
          prof.sync();
          if (n_queued > queue_capacity_) { n_queued = queue_capacity_; }
          if (n_queued > 0) {
            interactions_queued_ += n_queued;
            if (n_queued > max_queued_per_launch_) { max_queued_per_launch_ = n_queued; }
          }
          if (d_pending == nullptr) {
            drain_interactions(d_queue_, n_queued, tracks_[nxt].view, had_wiring, books);
          } else {
            // P34 (V294): drain now when the ring and this iteration's entries make a chunk of
            // the card's width, when the oldest pending entry has waited `defer` iterations, or
            // when they would not fit the ring; otherwise this iteration's entries wait.
            const int total = pending_n + n_queued;
            const bool fits = total <= pending_cap;
            // P36: a full-width drain is the flat pool's width when it runs the Binary bucket.
            const int width_ = flat_pool_().ok()
                                   ? std::max(n_interaction_slots_,
                                              std::min(flat_pool_().lanes, flat_wide_cap_()))
                                   : n_interaction_slots_;
            const bool wide = total >= width_;
            const bool aged = pending_n > 0 && iteration - pending_from >= defer;
            if (wide || aged || !fits) {
              if (pending_n == 0) {
                drain_interactions(d_queue_, n_queued, tracks_[nxt].view, had_wiring, books);
              } else if (fits) {
                if (n_queued > 0) {
                  G4GPU_CUDA_CHECK(cudaMemcpyAsync(
                      d_pending + pending_n, d_queue_,
                      sizeof(had::PendingInteraction<real_t>) * n_queued,
                      cudaMemcpyDeviceToDevice));
                  pending_n += n_queued;
                }
                drain_pending(tracks_[nxt].view, had_wiring, books);
              } else {
                drain_pending(tracks_[nxt].view, had_wiring, books);
                drain_interactions(d_queue_, n_queued, tracks_[nxt].view, had_wiring, books);
              }
            } else if (n_queued > 0) {
              if (pending_n == 0) { pending_from = iteration; }
              G4GPU_CUDA_CHECK(cudaMemcpyAsync(
                  d_pending + pending_n, d_queue_,
                  sizeof(had::PendingInteraction<real_t>) * n_queued, cudaMemcpyDeviceToDevice));
              pending_n += n_queued;
              deferred_entries += n_queued;
            }
          }
        } else if (pending_n > 0 && iteration - pending_from >= defer) {
          // No hadron stepped this iteration, so nothing was queued; the ring still ages.
          drain_pending(tracks_[nxt].view, had_wiring, books);
        }

        // ---- P19: drain the photo- and lepto-nuclear queue, ON THE DEVICE.
        //
        // After P15's drain, on the same stream, so the two never hold a slot at once and a
        // cascade's secondaries and a photon's land in the same `tracks_[nxt]` to be stepped next
        // iteration. Each launch is `n_slots` threads that read the cursor themselves: nothing
        // is copied back and the host does not wait (see `run_emextra_drain`), which is why this
        // runs on every iteration a photon, an e+- or a muon was stepped and is not gated on a
        // host-side count it could only get by synchronising. A launch whose species could not
        // have queued anything - no photon, or the process off - is skipped outright.
        if (any_emx && d_emx_queue_ != nullptr && interaction_pool_.ok) {
          // NO WIDER THAN THE SLOTS (P34, docs/RISK.md V294). Thread `i` takes slot `i`, so a
          // thread past the last slot finds none and refuses its entry as
          // `kInteractionNoSlot`. The grid was the slot count rounded UP to 32 - exact at P19's
          // 128 slots, and since V285 sized the pool to the card, 640 threads over 628 slots:
          // twelve threads with no slot, reached by an entry at a queue index past 627. No
          // iteration of the four beams queues that many photo-/lepto-nuclear interactions,
          // so the synchronous drain never met it; P32's asynchronous drain put them behind
          // thousands of hadronic entries in one ring, and THAT is the 201 track-steps it lost
          // on the pion beam - 4-5 photonuclear refusals a run, on the ledger. Rounded DOWN, the
          // grid is at most the slot count, and every thread has one.
          int kEth = 32;
          if (n_interaction_slots_ < kEth) { kEth = std::max(1, n_interaction_slots_); }
          const int eblocks = std::max(1, n_interaction_slots_ / kEth);
          if (nsp[kSpeciesGamma] > 0 && had_wiring.photon_nuclear) {
            prof.begin(LoopProfile::kEmxBase);
            run_emextra_drain<real_t, had::InteractionBucket::kPhotoNuclear>
                <<<eblocks, kEth>>>(scene_, d_emx_queue_, d_emx_cursor_, emx_queue_capacity_,
                                    tracks_[nxt].view, stride, d_win_score, d_voxel_score_,
                                    had_wiring, interaction_pool_.d_pool, traj, d_status_warn_, sec_,
                                    books, hook_);
            ++emx_drain_launches_;
            prof.end(LoopProfile::kEmxBase, 0);
            prof.launch();
          }
          const bool lepto_live =
              ((nsp[kSpeciesElectron] > 0 || nsp[kSpeciesPositron] > 0)
               && had_wiring.electro_nuclear)
              || ((nsp[kSpeciesMuonMinus] > 0 || nsp[kSpeciesMuonPlus] > 0)
                  && had_wiring.muon_nuclear);
          if (lepto_live) {
            prof.begin(LoopProfile::kEmxBase + 1);
            run_emextra_drain<real_t, had::InteractionBucket::kLeptoNuclear>
                <<<eblocks, kEth>>>(scene_, d_emx_queue_, d_emx_cursor_, emx_queue_capacity_,
                                    tracks_[nxt].view, stride, d_win_score, d_voxel_score_,
                                    had_wiring, interaction_pool_.d_pool, traj, d_status_warn_, sec_,
                                    books, hook_);
            ++emx_drain_launches_;
            prof.end(LoopProfile::kEmxBase + 1, 0);
            prof.launch();
          }
          G4GPU_CUDA_CHECK(cudaGetLastError());
        }

        prof.close_iteration(iteration, n_live, tracks_[cur].view.capacity, total_step);
        cur ^= 1;
        nxt ^= 1;
      }
      // P34 (V294): a window that ended on the iteration bound with interactions still pending
      // runs them, so their deposits are scored, and abandons what they give back with the rest.
      if (pending_n > 0) {
        const int before = tracks_[cur].count();
        drain_pending(tracks_[cur].view, last_wiring, last_books);
        leftover += tracks_[cur].count() - before;
      }
      st.abandoned += leftover;

      // Did this attempt lose anybody? Checked before a single number is accumulated.
      const long long dropped =
          tracks_[0].overflow() + tracks_[1].overflow();
      if (dropped > 0) {
        // Below this a batch is too small to be worth splitting further, and something other
        // than batch size is wrong - a single event whose shower cannot fit the whole arena.
        constexpr int kMinBatch = 256;
        if (try_batch > kMinBatch && !allow_dropped_) {
          try_batch /= 2;
          ++st.batch_retries;
          st.track_steps = steps_before;
          st.abandoned = abandoned_before;
          std::printf("batch %d..%d ran out of track slots; retrying %d events at a time\n",
                      base, base + n_batch, try_batch);
          continue;  // same events, smaller batch, nothing accumulated
        }
        if (!allow_dropped_) {
          std::printf("\n*** %lld TRACKS WERE DROPPED ***\n\n"
                      "    A single batch of %d events could not fit the track arena even\n"
                      "    after halving it down from %d. One event's shower needs more live\n"
                      "    slots than the whole pool holds, so a smaller batch cannot help.\n"
                      "    Raise SetLiveTracksPerEvent (currently %.1f per event).\n\n",
                      dropped, n_batch, batch_, live_per_event_);
          std::exit(3);
        }
        st.overflow += dropped;
      }

      G4GPU_CUDA_CHECK(cudaMemcpy(h_score.data(), d_win_score,
                                  sizeof(double) * n_scorers_ * stride,
                                  cudaMemcpyDeviceToHost));
      // The carried-away ledger, brought back with the scores and appended to the run-length
      // vector. Accumulated on the host per event rather than reduced on the device because the
      // per-event number is the one that means anything - see RunStats::carried_away - and the
      // array is already being copied for the batch.
      if (d_win_carried != nullptr) {
        if (st.carried_away.empty()) { st.carried_away.assign(n_events, 0.0); }
        std::vector<double> h_nu(static_cast<size_t>(n_batch));
        G4GPU_CUDA_CHECK(cudaMemcpy(h_nu.data(), d_win_carried, sizeof(double) * n_batch,
                                    cudaMemcpyDeviceToHost));
        for (int e = 0; e < n_batch; ++e) {
          st.carried_away[base + e] = h_nu[e];
          st.carried_away_total += h_nu[e];
        }
      }
      // Event-major, so that a sink sees one event's scores together. Each accumulator still
      // sums over events in ascending order, so the totals are bit-identical to the
      // scorer-major loop this replaced - which matters, because those totals are what the
      // 0.09-sigma agreement with Geant4 is measured on.
      for (int e = 0; e < n_batch; ++e) {
        for (int sc = 0; sc < n_scorers_; ++sc) {
          const double v = h_score[static_cast<size_t>(sc) * stride + e];
          score_sum[sc] += v;
          score_sum_sq[sc] += v * v;
          ev_scores[sc] = v;
        }
        if (sink != nullptr) { sink->Event(base + e, ev_scores.data(), n_scorers_); }
      }
      base += n_batch;
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
    // P32's profile, and beside it what the kernels it timed can have resident: registers,
    // frame and blocks per SM at the block size each is launched with, as the driver reports
    // them for this card. The stack raise does not appear among the limits on purpose - the
    // driver reserves the frame for every thread the card can hold whatever is launched - so
    // what bounds an interaction launch is its registers and its width, and the width is the
    // chunk: at most `n_interaction_slots_` threads, in blocks of 32.
    if (prof.on) {
      prof.report(ms);
      struct KernelRow {
        const char* name;
        const void* fn;
        int block;
      };
      const KernelRow rows[] = {
          {"run_step_gamma", reinterpret_cast<const void*>(run_step_gamma<real_t, StepHook>),
           threads_},
          {"run_step_lepton<e->",
           reinterpret_cast<const void*>(run_step_lepton<real_t, false, StepHook>), threads_},
          {"run_step_hadron<kProton>",
           reinterpret_cast<const void*>(run_step_hadron<real_t, ParticleType::kProton, StepHook>),
           threads_},
          {"run_step_hadron<kPionPlus>",
           reinterpret_cast<const void*>(
               run_step_hadron<real_t, ParticleType::kPionPlus, StepHook>),
           threads_},
          {"run_step_hadron<kGenericIon>",
           reinterpret_cast<const void*>(
               run_step_hadron<real_t, ParticleType::kGenericIon, StepHook>),
           threads_},
          {"run_step_neutral<kNeutron>",
           reinterpret_cast<const void*>(
               run_step_neutral<real_t, ParticleType::kNeutron, StepHook>),
           threads_},
          {"run_interaction<kFtfp>",
           reinterpret_cast<const void*>(
               run_interaction<real_t, had::InteractionBucket::kFtfp, StepHook>),
           1},
          {"run_interaction<kBertini>",
           reinterpret_cast<const void*>(
               run_interaction<real_t, had::InteractionBucket::kBertini, StepHook>),
           1},
          {"run_interaction<kBinary>",
           reinterpret_cast<const void*>(
               run_interaction<real_t, had::InteractionBucket::kBinary, StepHook>),
           1},
          {"run_interaction<kLightIon>",
           reinterpret_cast<const void*>(
               run_interaction<real_t, had::InteractionBucket::kLightIon, StepHook>),
           1},
          {"run_interaction<kAtRest>",
           reinterpret_cast<const void*>(
               run_interaction<real_t, had::InteractionBucket::kAtRest, StepHook>),
           1},
      };
      const host::SelectedDevice& dev = host::select_device();
      std::printf("  %-30s %6s %10s %6s %14s  (%d SMs, %d threads an SM; %d interaction "
                  "slots)\n", "kernel", "regs", "frame B", "block", "blocks/SM (warps)",
                  dev.prop.multiProcessorCount, dev.prop.maxThreadsPerMultiProcessor,
                  n_interaction_slots_);
      for (const KernelRow& r : rows) {
        cudaFuncAttributes a{};
        int per_sm = 0;
        if (cudaFuncGetAttributes(&a, r.fn) != cudaSuccess
            || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, r.fn, r.block, 0)
                   != cudaSuccess) {
          cudaGetLastError();
          continue;
        }
        std::printf("  %-30s %6d %10zu %6d %8d (%3d)\n", r.name, a.numRegs, a.localSizeBytes,
                    r.block, per_sm, per_sm * ((r.block + 31) / 32));
      }
    }
    prof.release();
    if (d_sort_summary != nullptr) { cudaFree(d_sort_summary); }
    if (h_sort_summary != nullptr) { cudaFreeHost(h_sort_summary); }
    if (d_pending != nullptr) {
      if (deferred_drains > 0) {
        std::printf("deferred drain: %lld interactions waited, drained in %lld launches of the "
                    "pending ring (at most %d iterations, %d entries; docs/RISK.md V294)\n",
                    deferred_entries, deferred_drains, defer, pending_cap);
      }
      cudaFree(d_pending);
    }
    // The window's own arrays, when it was wider than the batch; the batch-sized ones Upload
    // made are the engine's and stay.
    if (win_owned) {
      cudaFree(d_win_score);
      if (d_win_carried != d_carried_away_) { cudaFree(d_win_carried); }
      cudaFree(d_win_prim);
    }
    if (refills > 0) {
      std::printf("refill: %lld events seeded into freed pool slots in %lld refills (windows of "
                  "%d events, %d in flight; docs/RISK.md V284)\n",
                  refilled_events, refills, win_cap, batch_);
    }
    // st.overflow is accumulated per attempt above, and is only ever non-zero when a caller
    // asked for best-effort transport with AllowDroppedTracks. Reading the counters here
    // instead would report the last attempt's, which is zero by construction.
    if (d_status_warn_ != nullptr) {
      int warn = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&warn, d_status_warn_, sizeof(int), cudaMemcpyDeviceToHost));
      st.unsupported_track_status = warn;
    }
    if (st.throttled > 0) {
      std::printf("throttled: %lld track-steps deferred to a later iteration (pool %.1f per "
                  "event)\n", st.throttled, live_per_event_);
    }
    if (d_unknown_ != nullptr) {
      int unk = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&unk, d_unknown_, sizeof(int), cudaMemcpyDeviceToHost));
      if (unk > 0) {
        std::printf("\n*** %d TRACKS HAVE NO STEPPING KERNEL ***\n"
                    "    Their species is not in species_index, so they were never\n"
                    "    dispatched and their energy was never deposited. BufferEmitter::push\n"
                    "    refuses such a species before appending it and BeamOn refuses it as a\n"
                    "    primary, so this is a hole in one of those two guards rather than a\n"
                    "    particle to be accounted for.\n\n", unk);
      }
    }

    // ---- the three ledgers.
    //
    // Read back and reported here, together, because the three of them are one statement about
    // the run: of the energy that did not reach a scorer, this much left as neutrinos, this
    // much was deleted by the neutron time cut, and these particles were not transported at
    // all. Reported only when non-zero, so a gamma run's output is unchanged.
    {
      const int kNT = static_cast<int>(ParticleType::kNumTypes);
      std::vector<int> n_carried(kNT, 0), n_refused(kNT, 0);
      if (d_carried_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(n_carried.data(), d_carried_n_, sizeof(int) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      if (d_refused_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(n_refused.data(), d_refused_, sizeof(int) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      std::vector<double> e_refused(kNT, 0.0);
      if (d_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(e_refused.data(), d_refused_e_, sizeof(double) * kNT,
                                    cudaMemcpyDeviceToHost));
      }
      for (int t = 0; t < kNT; ++t) {
        st.carried_by_species[t] = n_carried[t];
        st.carried_away_n += n_carried[t];
        st.refused_by_species[t] = n_refused[t];
        st.refused_total += n_refused[t];
        st.refused_energy_by_species[t] = e_refused[t];
        st.refused_energy_total += e_refused[t];
      }
      if (d_killed_n_ != nullptr) {
        int kn = 0;
        G4GPU_CUDA_CHECK(cudaMemcpy(&kn, d_killed_n_, sizeof(int), cudaMemcpyDeviceToHost));
        st.neutron_killed_n = kn;
      }
      if (d_killed_energy_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(&st.neutron_killed_energy, d_killed_energy_, sizeof(double),
                                    cudaMemcpyDeviceToHost));
      }

      if (st.carried_away_n > 0) {
        std::printf("carried away: %lld particles, %.6g MeV total (not deposited, not"
                    " transported)\n", st.carried_away_n, st.carried_away_total);
        for (int t = 0; t < kNT; ++t) {
          if (n_carried[t] > 0) {
            std::printf("              %-12s %d\n",
                        particle_name(static_cast<ParticleType>(t)), n_carried[t]);
          }
        }
      }
      if (st.neutron_killed_n > 0) {
        // Not a warning. Geant4 does exactly this and also does not conserve energy across it;
        // the number is printed so that a run can be held to an energy balance rather than
        // leaving the shortfall to be found later and blamed on the transport.
        std::printf("neutron time cut: %lld neutrons killed past 10 us, %.6g MeV discarded\n"
                    "                  (G4NeutronGeneralProcess deposits nothing here - see"
                    " step_neutral)\n",
                    st.neutron_killed_n, st.neutron_killed_energy);
      }
      // P22: WHICH ORDER THE NEUTRON'S STEPS WERE CHARGED IN, on every run where it could act -
      // the final stage, the combined table on the device, a neutron stepped. 41% of a 100 MeV
      // neutron's B1 dose moves with it (docs/RISK.md V219), so a neutron number printed
      // without this line is not a number anyone can attribute.
      st.neutron_order = neutron_order_used_;
      st.neutron_order_acted = (had_stage_used_ == had::HadronicStage::kFinal
                                && d_neutron_xs_ != nullptr && neutron_iterations_ > 0);
      if (st.neutron_order_acted) {
        std::printf("neutron general process: %s order, a neutron alive in %lld of %lld "
                    "iterations - %s\n",
                    had::neutron_general_order_key(neutron_order_used_), neutron_iterations_,
                    iterations_run_, had::neutron_general_order_name(neutron_order_used_));
      }
      if (st.refused_total > 0) {
        // This one IS a warning, and it names what is missing. A refused secondary is energy
        // the shower had and this port did not carry, so the dose is low by whatever those
        // particles would have deposited - the same failure mode as a dropped track, arrived at
        // for a different reason.
        std::printf("\n*** %lld SECONDARIES OF SPECIES THIS PORT CANNOT TRANSPORT, %.6g MeV"
                    " ***\n\n"
                    "    They were counted at the point a process created them and no track\n"
                    "    was made, so the dose from this run is TOO LOW by whatever they\n"
                    "    would have deposited. The energy column is what the answer is\n"
                    "    missing; the count alone does not say.\n\n",
                    st.refused_total, st.refused_energy_total);
        for (int t = 0; t < kNT; ++t) {
          if (n_refused[t] > 0) {
            std::printf("      %-12s %10lld   %12.6g MeV\n",
                        particle_name(static_cast<ParticleType>(t)),
                        st.refused_by_species[t], st.refused_energy_by_species[t]);
          }
        }
        std::printf("\n");
      }
    }

    // ---- P8's per-process refusals.
    //
    // Separate from the species ledger above and NOT foldable into it: that one is "this
    // transport has no kernel for that particle", this one is "this transport reached that
    // PROCESS and has no final state for it". A stopped pi- is not a refused species - pi- has
    // a kernel and was transported all the way to rest - it is a refused capture, and the
    // energy it costs is a rest mass rather than a kinetic one.
    {
      const int kNR = static_cast<int>(had::HadronicRefusal::kNumHadronicRefusals);
      std::vector<int> hn(kNR, 0);
      std::vector<double> he(kNR, 0.0);
      if (d_had_refused_n_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(hn.data(), d_had_refused_n_, sizeof(int) * kNR,
                                    cudaMemcpyDeviceToHost));
      }
      if (d_had_refused_e_ != nullptr) {
        G4GPU_CUDA_CHECK(cudaMemcpy(he.data(), d_had_refused_e_, sizeof(double) * kNR,
                                    cudaMemcpyDeviceToHost));
      }
      for (int r = 0; r < kNR; ++r) {
        st.had_refused_count[r] = hn[r];
        st.had_refused_energy[r] = he[r];
        st.had_refused_total += hn[r];
      }
      if (st.had_refused_total > 0) {
        std::printf("\n*** %lld HADRONIC INTERACTIONS WITH NO FINAL STATE IN THIS PORT ***\n\n"
                    "    Stage: %s\n"
                    "    The cross section decided each of these happened; the model that\n"
                    "    would have said what came out is not written. The dose is low by the\n"
                    "    energy column, which is what the missing model would have moved.\n"
                    "\n"
                    "    SINCE P15 THE LEDGER HAS TWO GROUPS AND THEY MUST NOT BE ADDED.\n"
                    "    The first says HOW MUCH is missing - one entry per lost interaction,\n"
                    "    with the projectile's kinetic energy on it. The second says WHY, and\n"
                    "    every one of its entries is a second booking on an event that is\n"
                    "    already in the first. Sum the first group, read the second.\n\n",
                    st.had_refused_total, had::hadronic_stage_name(had_stage_used_));
        for (int r = 0; r < kNR; ++r) {
          if (hn[r] > 0) {
            std::printf("      %10lld   %12.6g MeV   %s\n", st.had_refused_count[r],
                        st.had_refused_energy[r],
                        had::hadronic_refusal_name(static_cast<had::HadronicRefusal>(r)));
          }
        }
        std::printf("\n");
      }
    }
    // ---- P15: the interaction queue and the pool, whether or not anything was refused.
    //
    // PRINTED EVEN WHEN NOTHING WENT WRONG, because the whole argument for the slot count is
    // that it is a measured capacity rather than an implicit one: a run that does not say how
    // many interactions it produced and how many launches it took to run them has not made the
    // capacity visible, it has only made it invisible in a different place.
    if (flat_stats_().entries > 0) {
      const FlatStats& fs = flat_stats_();
      std::printf("flat Binary cascade: %lld entries, %lld finished on %d lanes in %lld rounds "
                  "(%lld launches); handed to the thread path: %lld straggler, %lld capacity, "
                  "%lld refused, %lld hydrogen, %lld PreCompound, %lld re-entry; %lld tail waits "
                  "(docs/RISK.md V309)\n",
                  fs.entries, fs.finished, flat_pool_().lanes, fs.rounds, fs.launches,
                  fs.handed[1], fs.handed[2], fs.handed[3], fs.handed[4], fs.handed[5],
                  fs.handed[6], fs.tail_waits);
    }
    if (flat_ion_stats_().entries > 0) {
      const FlatStats& fs = flat_ion_stats_();
      std::printf("flat light-ion reaction: %lld entries, %lld finished on %d lanes in %lld rounds "
                  "(%lld launches); handed to the thread path: %lld straggler, %lld capacity, "
                  "%lld refused, %lld fusion, %lld re-entry; %lld tail waits "
                  "(docs/RISK.md V310)\n",
                  fs.entries, fs.finished, flat_ion_pool_().lanes, fs.rounds, fs.launches,
                  fs.handed[1], fs.handed[2], fs.handed[3], fs.handed[5], fs.handed[6],
                  fs.tail_waits);
    }
    st.interactions = interactions_queued_;
    st.interaction_chunks = interaction_chunks_;
    st.max_queued_per_launch = max_queued_per_launch_;
    st.interaction_slots = n_interaction_slots_;
    st.interaction_bytes = interaction_bytes_;
    if (interactions_queued_ > 0) {
      std::printf("interactions: %lld queued in %lld chunk launches over %d slots (%.1f MB); "
                  "the busiest iteration queued %d (%.1f%% of the %d-entry queue)\n",
                  interactions_queued_, interaction_chunks_, n_interaction_slots_,
                  double(interaction_bytes_) / 1048576.0, max_queued_per_launch_,
                  100.0 * double(max_queued_per_launch_)
                      / double(queue_capacity_ ? queue_capacity_ : 1),
                  queue_capacity_);
      if (interactions_no_model_ > 0) {
        std::printf("             %lld of them had no model in range and never reached a "
                    "kernel (kNoInelasticModel above)\n", interactions_no_model_);
      }
    }
    // ---- P19: the photo- and lepto-nuclear queue, read back ONCE, here.
    //
    // Printed whenever the run could have queued anything, zero included: a photon beam that
    // reports "0 photonNuclear" has said something - that its shower never reached a nucleus -
    // and one that prints nothing has said nothing. The third number is RISK V207's finding,
    // counted on this run's photons.
    if (d_emx_stats_ != nullptr && emx_drain_launches_ > 0) {
      unsigned long long es[kEmxCounters] = {};
      G4GPU_CUDA_CHECK(cudaMemcpy(es, d_emx_stats_, sizeof(es), cudaMemcpyDeviceToHost));
      st.photo_nuclear = static_cast<long long>(es[had::kEmxPhotoDrained]);
      st.lepto_nuclear = static_cast<long long>(es[had::kEmxLeptoDrained]);
      st.photo_nuclear_to_conversion = static_cast<long long>(es[had::kEmxZone3Conversion]);
      st.emx_max_queued = static_cast<long long>(es[kEmxQueueDepth]);
      std::printf("photo-/lepto-nuclear: %llu photonNuclear and %llu electron-/positron-/"
                  "muonNuclear interactions drained on the device in %lld launches; %llu "
                  "photon interactions >= 100 MeV whose photonNuclear share went to conversion "
                  "(G4GammaGeneralProcess 11.1.1, docs/RISK.md V207)%s; the deepest iteration "
                  "queued %llu of the %d-entry queue; a hadron was alive - P15's read-back - in "
                  "%lld of %lld iterations\n",
                  es[had::kEmxPhotoDrained], es[had::kEmxLeptoDrained], emx_drain_launches_,
                  es[had::kEmxZone3Conversion],
                  emx_config_used_[3] ? "" : " [general process OFF]", es[kEmxQueueDepth],
                  emx_queue_capacity_, hadron_iterations_, iterations_run_);
      // P21: the photo-nuclear interactions table 9 made where the store's cross section is
      // exactly zero. Printed only when there were any: zero is the expectation of almost every
      // run, and it is in the line above's photonNuclear count either way.
      if (es[had::kEmxTable9Edge] > 0) {
        std::printf("photo-/lepto-nuclear: %llu of the photonNuclear interactions were table 9's "
                    "edge - selected by G4GammaGeneralProcess's linear table where the "
                    "photo-nuclear cross section is exactly zero, and applied to the material's "
                    "first element as Geant4's SampleZandA applies them\n",
                    es[had::kEmxTable9Edge]);
      }
      // THE STACK THE DRAINS RAN UNDER. They never raise it themselves; if the driver had to,
      // the limit says so - the same tripwire as P15's below, for a kernel that launches on
      // photon runs where nothing else would ever have raised it.
      if (!stack_raised_) {
        std::size_t lim = 0;
        G4GPU_CUDA_CHECK(cudaDeviceGetLimit(&lim, cudaLimitStackSize));
        if (lim > stepping_stack_bytes()) {
          std::printf("STACK: the driver raised the device stack to %zu B at a photo-/lepto-"
                      "nuclear drain launch (frames %zu and %zu B) - docs/RISK.md V196, V210.\n",
                      lim, emx_frame_photo_, emx_frame_lepto_);
        }
      }
    }
    emx_drain_launches_ = 0;
    // THE TRIPWIRE ON THE STACK RESERVATION. If the driver found the limit short at an
    // interaction launch it raised it itself (V196) - no fault, no error code, and 270 MB of an
    // 8 GB card per 4 kB spent after the pools were sized against the free memory printed at
    // the raise. The only trace it leaves is the limit, so the limit is read here and compared.
    // A run with no hadron never raised anything and has nothing to compare.
    if (stack_raised_) {
      std::size_t lim = 0;
      G4GPU_CUDA_CHECK(cudaDeviceGetLimit(&lim, cudaLimitStackSize));
      if (lim > stack_reserved_) {
        std::printf("STACK: the driver raised the device stack from the %zu B reserved off %s to "
                    "%zu B at an interaction launch - the reservation read off the kernels was "
                    "short. docs/RISK.md V196.\n", stack_reserved_, stack_kernel_, lim);
      } else {
        std::printf("stack: %zu B a thread reserved off %s, and no launch needed more\n",
                    stack_reserved_, stack_kernel_);
      }
    }
    if (sec_.overflow != nullptr) {
      int over = 0;
      G4GPU_CUDA_CHECK(cudaMemcpy(&over, sec_.overflow, sizeof(int), cudaMemcpyDeviceToHost));
      st.secondary_overflow = over;
      // PRINTED, WHICH IT WAS NOT.
      //
      // The count has been recorded on RunStats since the arena was written, and the field's
      // own comment says "Zero in every run of this project's pipeline; if it is not zero,
      // raise the arena" - which is advice to whoever reads the field, and until now the only
      // thing that read it was tests/test_custom_hook.cu. g4dose, example B1, the viewer, the
      // builder and every generated project ignored it, so an arena that overflowed said
      // nothing at all.
      //
      // What overflow costs is narrow and worth stating exactly, because it is NOT a wrong
      // dose: the arena holds one (buffer, slot) pair per secondary so that a stepping action
      // can walk this step's children, and a pair that does not fit only breaks that walk.
      // Every track is still in the pool and every deposit is still scored. So this is a
      // warning about a hook's input, not about the physics - which is precisely why it needs
      // printing rather than exiting, and precisely why it could sit unread: nothing downstream
      // gets worse in a way anyone would notice.
      //
      // The default capacity is now the whole pool rather than half the batch, so overflow
      // needs the output buffer to have overflowed first - and that ends the run. It is
      // reachable only through SetSecondaryArenaCapacity.
      if (over > 0) {
        std::printf("\n*** %d SECONDARY CHAIN ENTRIES DID NOT FIT THE ARENA ***\n\n"
                    "    The arena holds %d entries. Every track was still appended to the\n"
                    "    pool and every deposit was scored, so the DOSE IS UNAFFECTED - what\n"
                    "    is incomplete is DeviceStep::sec_last, the chain a stepping action\n"
                    "    walks to see this step's secondaries. A hook that reads it saw fewer\n"
                    "    children than the step made. Raise it with\n"
                    "    SetSecondaryArenaCapacity(n) - the default is the whole track pool.\n\n",
                    over, sec_.capacity);
      }
    }
    // A dropped track is a track whose energy never reached a scorer, so the dose this run
    // reports is too low by whatever it would have deposited - and the run otherwise finishes
    // normally and prints a number that looks exactly like a result.
    //
    // So this ends the run. "Report it and carry on" is not defensible in a code whose whole
    // claim is agreeing with Geant4 to fractions of a sigma: a quietly-low dose is worse than
    // no dose, because someone will use it. Until today the only thing that looked at this at
    // all was example B1's RunAction - g4dose, the viewer, the builder and every generated
    // project ignored it, and would have reported a short answer as a result.
    //
    // AllowDroppedTracks() exists for the case where somebody genuinely wants best-effort
    // transport and has decided the shortfall is acceptable. It has to be asked for.
    if (st.overflow > 0) {
      std::printf("\n*** %d TRACKS WERE DROPPED ***\n\n"
                  "    A species buffer had no room, so those particles stopped being\n"
                  "    transported. Their energy was never deposited, so the dose from this\n"
                  "    run is TOO LOW - it is not a measurement.\n\n"
                  "    Raise SetLiveTracksPerEvent (currently %.1f per event) or lower the\n"
                  "    batch. Call AllowDroppedTracks(true) if a short answer is genuinely\n"
                  "    what you want.\n\n",
                  st.overflow, live_per_event_);
      if (!allow_dropped_) { std::exit(3); }
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return st;
  }



template <typename real_t, typename StepHook>
void TransportEngine<real_t, StepHook>::Free() {
    for (int i = 0; i < 2; ++i) {
      tracks_[i].free_all();
      cudaFree(idx_[i]);
      cudaFree(idx_n_[i]);
      idx_[i] = nullptr;
      idx_n_[i] = nullptr;
    }
    cudaFree(d_unknown_);
    d_unknown_ = nullptr;
    cudaFree(d_carried_away_);
    cudaFree(d_carried_n_);
    cudaFree(d_refused_);
    cudaFree(d_refused_e_);
    cudaFree(d_had_refused_n_);
    cudaFree(d_had_refused_e_);
    cudaFree(d_killed_energy_);
    cudaFree(d_killed_n_);
    d_carried_away_ = nullptr;
    d_carried_n_ = nullptr;
    d_refused_ = nullptr;
    d_refused_e_ = nullptr;
    d_had_refused_n_ = nullptr;
    d_had_refused_e_ = nullptr;
    d_killed_energy_ = nullptr;
    d_killed_n_ = nullptr;
    // NOT a cudaFree of d_neutron_xs_: it points INTO neutron_tables_, which owns every
    // allocation behind it, so freeing both would be a double free. It used to be its own
    // allocation - and it used to be permanently null, which is why nobody noticed.
    d_neutron_xs_ = nullptr;
    free_neutron_tables<real_t>(neutron_tables_);
    free_elastic_tables<real_t>(elastic_tables_);
    free_inelastic_tables<real_t>(inelastic_tables_);
    free_emextra_tables<real_t>(emextra_tables_);
    free_level_data(level_tables_);
    // P15's pool through its contract (docs/RISK.md V233), then the queue.
    had::free_interaction_pool<real_t>(interaction_pool_);
    cudaFree(d_queue_);
    cudaFree(d_queue_cursor_);
    cudaFree(d_qidx_);
    cudaFree(d_qcount_);
    cudaFree(d_emx_queue_);
    cudaFree(d_emx_cursor_);
    cudaFree(d_emx_stats_);
    d_emx_stats_ = nullptr;
    d_queue_ = nullptr;
    d_queue_cursor_ = nullptr;
    d_qidx_ = nullptr;
    d_qcount_ = nullptr;
    d_emx_queue_ = nullptr;
    d_emx_cursor_ = nullptr;
    cudaFree(d_vols_);
    cudaFree(d_mats_);
    cudaFree(d_rt_);
    cudaFree(d_wv_);
    cudaFree(d_score_);
    cudaFree(d_status_warn_);
    d_status_warn_ = nullptr;
    cudaFree(d_track_arena_);
    d_track_arena_ = nullptr;
    cudaFree(d_dropped_);
    cudaFree(d_dropped_count_);
    d_dropped_ = nullptr;
    d_dropped_count_ = nullptr;
    cudaFree(sec_.entry);
    cudaFree(sec_.prev);
    cudaFree(sec_.cursor);
    cudaFree(sec_.overflow);
    sec_ = SecondaryArena{};
    cudaFree(d_primaries_);
    d_primaries_ = nullptr;
    cudaFree(d_voxel_score_);
    d_voxel_score_ = nullptr;
    n_voxel_cells_ = 0;
    cudaFree(d_pool_solids_);
    cudaFree(d_pool_xforms_);
    cudaFree(d_pool_aux_);
    cudaFree(d_voxels_);
    cudaFree(d_voxel_class_);
    cudaFree(d_class_layer_);
    cudaFree(d_class_absent_);
    cudaFree(d_tri_);
    cudaFree(d_bvh_);
  }

}  // namespace g4gpu::host
