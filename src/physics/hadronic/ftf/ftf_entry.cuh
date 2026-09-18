// The ONE header a caller outside this package includes to run FTFP.
//
// `ftf::apply_yourself` needs a 400-kilobyte workspace and a small table, and until P11d nothing
// but `tests/test_ftf_model.cu` knew how to build either - so P12's `G4HadronicAbsorptionFritiof`
// arm refused 200,000 at-rest captures one level above the model, and P13's FTFP photon arm was
// about to do the same. This header is the fix, and it exposes exactly four things:
//
//   entry::Workspace / entry::HadronWorkspace   the two sized workspace types, with their bytes
//   entry::Handle<WS>                           what a kernel thread takes one slot of
//   entry::build / entry::free                  HOST-ONLY: allocate, initialise, upload, release
//   entry::apply                                the call, and a report a caller can act on
//
// Nothing of the model is re-exported. A caller that wants `FtfModelWorkspace`, `ExcitedString`,
// `FtfParameters` or any of the four dozen `ftf_*` functions has to include the model's own
// headers and is then not a caller any more.
//
// ## PER THREAD IN FLIGHT, NOT PER TRACK AND NOT PER EVENT
//
// The workspace lives for the duration of ONE `apply_yourself` call. A track that is not inside
// that call needs nothing, so the number of workspaces a run needs is the number of threads that
// can be INSIDE IT AT ONCE - not the batch, not the track pool, not the event count. That
// distinction is the whole of the sizing problem, because the two numbers differ by four orders
// of magnitude:
//
//   entry::Workspace       410,824 B   ion beams included (kMaxProjA = 64)
//   entry::HadronWorkspace 266,344 B   projectile is a single hadron (kMaxProjA = 1)
//
//   slots      Workspace      HadronWorkspace
//       64       26.3 MB           17.0 MB
//      256      105.2 MB           68.2 MB
//    1,024      420.7 MB          272.8 MB
//   65,536       26.9 GB           17.5 GB     <- what a per-TRACK reading would cost
//
// A 65,536-track batch does not need 65,536 workspaces and could not have them: 26.9 GB is past
// every card this project targets. What it needs is as many as the launch runs concurrently, and
// since a thread cannot portably learn its own residency, the contract is the other way round:
// **the caller states `n_slots`, a thread takes slot `tid`, and a thread whose `tid` is past the
// end is REFUSED BY NAME** (`kNoWorkspaceSlot`) rather than sharing a workspace with another
// thread. Sharing would be silent and wrong; refusing is loud and lets the caller either grow
// the pool or chunk the launch.
//
// For P12 that bound is small and knowable: the tracks that stop in one launch are a few per
// thousand, and the Fritiof arm sees only anti-baryons among them. 256 slots of
// `HadronWorkspace` is 68 MB and covers a batch in which 256 anti-nucleons come to rest at once.
//
// ## WHY THE UPLOADED IMAGE IS A CONSTRUCTED ONE AND NOT A cudaMemset
//
// docs/RISK.md V104: `FtfParameters` has members that only its CONSTRUCTOR sets, one of which
// decides whether a nucleus diffracts, and a zeroed workspace gets them wrong. So `build`
// default-constructs ONE host image, uploads it into every slot, and does not memset. The
// pointers that image carries - `Nucleus3DScratch::momentum` and the two `Nucleus3D::nucleons`,
// which point into the workspace's own arrays - are nulled before the upload: `ftf_model_init`
// re-wires all of them from the DEVICE workspace's address on every call, so they are never
// read before they are written, and a null faults at once where a stale host address would read
// whatever happened to be there.
//
// ## WHAT IS SHARED AND WHAT IS NOT
//
// `LundTables<double>` is 9,648 bytes, read-only, and identical for every track: ONE device copy
// for the whole run, in the handle. The big Lund data the fragmentation reads - 213 kB of it -
// is not here at all; it is `__device__` const data in the compiled unit, the way
// `data/ftf_hadrons.hh` is, and costs nothing per slot.
#pragma once
#include <cstddef>
#include <cstdio>

#include "physics/hadronic/ftf/theo_fs_generator.cuh"

namespace g4gpu::hadronic::ftf::entry {

/// The workspace for a run that can have an ION beam: `kMaxProjA = 64` covers every projectile a
/// galactic-cosmic-ray problem contains, and 410,824 bytes is what it costs. The template
/// arguments are the ones `test_ftf_model.cu`'s device probe is measured with, so PORTED
/// 2.1.11b's 255 registers and 864-byte frame are this type's numbers.
using Workspace = FtfWorkspace<250, 64, 1024, 512, 256, 96>;

/// The workspace for a run whose projectile is always a single hadron - P12's at-rest captures,
/// P13's photons, and any beam of p, n, pi, K, anti-nucleon or hyperon. `kMaxProjA = 1` removes
/// the projectile nucleus and its scratch; an ion handed to this one is refused by capacity
/// (`FtfModelReport::involved_capacity`, with `refused_a` naming the mass number), never
/// truncated.
using HadronWorkspace = FtfWorkspace<250, 1, 256, 256, 256, 96>;

/// What `apply` did, in the four terms a caller can act on.
enum class Status : int {
  kRan = 0,          ///< secondaries are in `out`
  kNoWorkspaceSlot,  ///< `slot_index` was past `n_slots`; grow the pool or chunk the launch
  kRefused,          ///< the model or the hand-over refused BY NAME; see `Report`
  /// `G4VPartonStringModel::Scatter` used all 1000 attempts and returned THE PRIMARY UNCHANGED
  /// at `z = 2*OuterRadius`, which is what Geant4 does there (with a JustWarning). It is its own
  /// status and not `kRefused` because a final state DID come back and a caller that treated it
  /// as "nothing happened" would double-count the primary. Measured: every proton, pi+ and alpha
  /// at 1 MeV lands here, because QBBC hands those to the cascades and not to FTFP - FTFP is
  /// failing to make an interaction it was never asked for. It is not free: see `apply`'s note
  /// on cost.
  kPrimaryUnchanged,
};

/// The detail behind the three answers that are not `kRan`, without exposing the model's own
/// report structs.
struct Report {
  FtfRefusal refused = FtfRefusal::kNone;  ///< the FTF-side refusal, `kNone` if not one
  bool generator_refused = false;          ///< P6's `Propagate` refused (short-lived track, ...)
  bool capacity = false;                   ///< a workspace capacity, not a physics gap
  bool attempts_exhausted = false;         ///< all 1000 Scatter attempts failed
  bool primary_unchanged = false;          ///< ...and the primary came back as the final state
  int attempts = 0;                        ///< G4VPartonStringModel::Scatter's retry count
  int n_secondaries = 0;
};

/// What a kernel thread takes one slot of. Copied BY VALUE into a launch, like
/// `had::HadronicWiring` and `had::ElasticTables` are (see `host/hadronic_upload.cuh`).
template <typename WS>
struct Handle {
  WS* slots = nullptr;
  const LundTables<double>* lund = nullptr;
  int n_slots = 0;

  /// The slot for thread @p i, or null when the caller under-provisioned. Never wraps: two
  /// threads in one workspace is a data race whose symptom is a wrong shower, and
  /// `Status::kNoWorkspaceSlot` is the alternative.
  __host__ __device__ WS* slot(int i) const {
    return (slots != nullptr && i >= 0 && i < n_slots) ? &slots[i] : nullptr;
  }
  __host__ __device__ bool ok() const {
    return slots != nullptr && lund != nullptr && n_slots > 0;
  }
};

/// Run one FTFP interaction.
///
/// @param h           the handle a launch was given by value
/// @param slot_index  which workspace to use; the global thread index is the intended value
/// @param proj        the projectile, as P5's framework carries it
/// @param target      (A, Z) of the target nucleus
/// @param out         the final state; cleared by `apply_yourself` itself
/// @param rep         what happened, when the answer is not `kRan`
///
/// The call is `__host__ __device__` because that is what the model is: the tests run it on the
/// host and `run_step_hadron` runs it on the device, from the same source.
///
/// COST, MEASURED ON THE HOST, per call at N = 20 (`tests/test_ftf_entry.cu` prints it):
///
///   in FTFP's window   proton 10 GeV on C 0.019 ms, on Pb 0.193 ms; alpha 10 GeV on Pb 0.234 ms
///   anti-proton 1 MeV  on C 0.015 ms, on Pb 0.162 ms   <- P12's case, and it RUNS
///   out of the window  proton/pi+/alpha at 1 MeV: 7 ms on C and 185-242 ms on Pb, ALL of it
///                      the 1000 Scatter attempts, none of which can succeed
///
/// The last line is the one a caller has to design around: it is 1,002 attempts every time, each
/// rebuilding both nuclei, and on a GPU it is one thread held for a quarter of a second. The fix
/// is not in the model - Geant4 spends the same attempts - it is for the caller to respect the
/// energy window `ref/oracle/ftf_windows.csv` records, which is what QBBC's own builders do.
template <typename real_t, int kMaxSec, typename WS, typename Rng>
__host__ __device__ inline Status apply(const Handle<WS>& h, int slot_index,
                                        const physics::hadronic::HadProjectile<real_t>& proj,
                                        const physics::hadronic::HadNucleus& target,
                                        physics::hadronic::HadFinalState<real_t, kMaxSec>& out,
                                        Report& rep, Rng& rng) {
  rep = Report();
  WS* ws = h.slot(slot_index);
  if (ws == nullptr || h.lund == nullptr) { return Status::kNoWorkspaceSlot; }

  // The report is per call and the workspace is reused, so it is cleared here rather than
  // trusted to be clean - the same reason `apply_yourself` clears `out`.
  ws->report = FtfApplyReport();
  apply_yourself(proj, target, out, ws, h.lund, rng);

  rep.attempts = ws->report.attempts;
  rep.n_secondaries = out.n_secondaries;
  rep.refused = (ws->report.refused != FtfRefusal::kNone) ? ws->report.refused
                                                          : ws->report.model.refused;
  rep.generator_refused = ws->report.generator.any();
  rep.capacity = ws->report.track_capacity || ws->report.secondary_overflow ||
                 ws->report.model.string_capacity || ws->report.model.involved_capacity ||
                 ws->report.model.additional_capacity;
  rep.attempts_exhausted = ws->report.attempts_exhausted;
  rep.primary_unchanged = ws->report.primary_returned_unchanged;
  // The order matters. `attempts_exhausted` comes with a final state - the primary - so it is
  // tested BEFORE the generic refusal, and `kRan` is reserved for a final state the model
  // actually built.
  if (rep.primary_unchanged || rep.attempts_exhausted) { return Status::kPrimaryUnchanged; }
  if (out.n_secondaries > 0 && !ws->report.any()) { return Status::kRan; }
  return Status::kRefused;
}

/// `FtfRefusal`'s own name function, re-exported so a caller need not include `refusal.cuh`.
__host__ __device__ inline const char* refusal_name(FtfRefusal r) {
  return ftf_refusal_name(r);
}

/// The device bytes @p n_slots of @p WS cost, plus the one shared table. Stated rather than
/// estimated: `sizeof` is the answer and the caller can ask before it allocates.
template <typename WS>
inline std::size_t bytes_for(int n_slots) {
  return sizeof(WS) * static_cast<std::size_t>(n_slots) + sizeof(LundTables<double>);
}

/// How many slots of @p WS fit in @p budget device bytes, at least one.
template <typename WS>
inline int slots_for_bytes(std::size_t budget) {
  if (budget <= sizeof(LundTables<double>)) { return 1; }
  const std::size_t n = (budget - sizeof(LundTables<double>)) / sizeof(WS);
  return (n < 1) ? 1 : static_cast<int>(n);
}

// ---------------------------------------------------------------------------------------------
// HOST ONLY from here down. These three functions call cudaMalloc/cudaMemcpy and must not be
// reached from device code; they are in this header rather than in `src/host/` so that a caller
// includes ONE file, which is the point of the header.
// ---------------------------------------------------------------------------------------------

/// Everything `build` allocated, so a run can give it back - the shape
/// `host/hadronic_upload.cuh`'s `ElasticTableOwner` has, for the same reason.
template <typename WS>
struct Owner {
  Handle<WS> view{};
  void* slots = nullptr;
  void* lund = nullptr;
  std::size_t bytes = 0;
};

/// Allocate, initialise and upload @p n_slots workspaces and the one shared Lund table.
///
/// @param n_slots  how many threads may be inside `apply` AT ONCE. See the header's table.
/// @param enable_bc_particles `G4HadronicParameters::EnableBCParticles`, 1 in 11.1.1 - it moves
///        the Lund flavour table's charm and bottom rows, so it is a parameter and not a
///        constant here.
/// @param verbose  prints the cost, as `upload_elastic_tables` and the range tables do.
///
/// Returns an owner whose `view.ok()` is false if anything failed; the caller decides whether
/// that is fatal. Nothing here calls exit().
template <typename WS>
inline Owner<WS> build(int n_slots, bool enable_bc_particles = true, bool verbose = true) {
  Owner<WS> own;
  if (n_slots < 1) { n_slots = 1; }

  // ---- the shared table. `lund_init` is the two G4VLongitudinalStringDecay constructors and
  // SetMinMasses; it reads nothing and allocates nothing, so a host instance is the whole of it.
  {
    auto* h = new LundTables<double>();
    lund_init(h, enable_bc_particles);
    if (cudaMalloc(&own.lund, sizeof(*h)) != cudaSuccess ||
        cudaMemcpy(own.lund, h, sizeof(*h), cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("FTF entry: could not upload %zu bytes of LundTables\n", sizeof(*h));
      delete h;
      cudaFree(own.lund);
      own.lund = nullptr;
      return own;
    }
    own.bytes += sizeof(*h);
    own.view.lund = static_cast<const LundTables<double>*>(own.lund);
    delete h;
  }

  // ---- the per-slot workspaces. ONE constructed host image, uploaded n_slots times.
  //
  // Not a cudaMemset: docs/RISK.md V104 is a member that only the constructor sets and that
  // decides physics when it is zero. Not n_slots host images either - 410 kB each would be
  // 420 MB of host memory for 1,024 slots and every one of them identical.
  {
    const std::size_t want = sizeof(WS) * static_cast<std::size_t>(n_slots);
    if (cudaMalloc(&own.slots, want) != cudaSuccess) {
      std::printf("FTF entry: could not allocate %zu bytes for %d workspace slots\n", want,
                  n_slots);
      cudaFree(own.lund);
      own.lund = nullptr;
      own.view = Handle<WS>{};
      return own;
    }
    auto* image = new WS();
    // The pointers the image carries point into the HOST copy's arrays. `ftf_model_init`
    // re-wires every one of them from the device workspace's own address before anything reads
    // them, so they are dead on arrival - and they are nulled so that a future caller which
    // reads one faults immediately instead of reading a host address as if it were a device one.
    image->model.scratch.momentum = nullptr;
    image->model.scratch.fermi_p = nullptr;
    image->model.scratch.test_sums = nullptr;
    image->model.scratch.flat_block = nullptr;
    image->model.scratch.capacity = 0;
    image->model.target.nucleons = nullptr;
    image->model.target.capacity = 0;
    image->model.projectile.nucleons = nullptr;
    image->model.projectile.capacity = 0;
    bool ok = true;
    for (int i = 0; i < n_slots && ok; ++i) {
      WS* dst = static_cast<WS*>(own.slots) + i;
      ok = (cudaMemcpy(dst, image, sizeof(WS), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    delete image;
    if (!ok) {
      std::printf("FTF entry: could not upload the workspace image into %d slots\n", n_slots);
      cudaFree(own.slots);
      cudaFree(own.lund);
      own.slots = nullptr;
      own.lund = nullptr;
      own.view = Handle<WS>{};
      return own;
    }
    own.bytes += want;
    own.view.slots = static_cast<WS*>(own.slots);
    own.view.n_slots = n_slots;
  }

  if (verbose) {
    std::printf("FTFP entry: %d workspace slots of %zu B + %zu B of tables = %.2f MB\n",
                n_slots, sizeof(WS), sizeof(LundTables<double>),
                double(own.bytes) / 1048576.0);
  }
  return own;
}

template <typename WS>
inline void free(Owner<WS>& own) {
  cudaFree(own.slots);
  cudaFree(own.lund);
  own = Owner<WS>{};
}

/// A handle over HOST memory, for a caller that runs the model on the host - which is what every
/// test of this package does. Same type, same `apply`, no cudaMalloc; the caller owns the array.
template <typename WS>
inline Handle<WS> host_handle(WS* slots, int n_slots, LundTables<double>* lund,
                              bool enable_bc_particles = true) {
  lund_init(lund, enable_bc_particles);
  Handle<WS> h;
  h.slots = slots;
  h.lund = lund;
  h.n_slots = n_slots;
  return h;
}

}  // namespace g4gpu::hadronic::ftf::entry
