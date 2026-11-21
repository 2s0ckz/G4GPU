// P38: the nucleus factory's host-side launch, declared for any translation unit; the kernels
// themselves are compiled once, in src/host/transport_run_npool_factory.cu (docs/RISK.md V319),
// so that a caller of the factory does not instantiate them again.
#ifndef G4GPU_NPOOL_NUCLEUS_FACTORY_LAUNCH_CUH
#define G4GPU_NPOOL_NUCLEUS_FACTORY_LAUNCH_CUH

#include <cuda_runtime.h>

namespace g4gpu::npool {

struct FactoryJob;
struct FactoryOutput;
struct FactoryRequest;

/// Samples `job.n` nuclei of `(job.a, job.z)` into `out`, a block of `factory_width(job.a)`
/// lanes per nucleus, the grid striding over the nuclei. `grid` 0 means one block a nucleus up
/// to 65,535. Returns the launch's error. The nuclei do not depend on `grid` (every draw is
/// keyed on the nucleus, the lane and the phase), which `tests/test_npool_nucleus.cu` checks.
cudaError_t launch_nucleus_factory(const FactoryJob& job, const FactoryOutput& out,
                                   cudaStream_t stream = 0, int grid = 0);

/// P39 (docs/RISK.md V321): the flat cascade's request lists - `*count32` requests of A <= 32 on
/// 32-lane blocks and `*count64` of 32 < A <= 64 on 64-lane blocks, the counts read on the
/// device. `max32` / `max64` bound the counts (the grids are sized from them, the blocks past
/// the count return at once). Both launches go on `stream`.
cudaError_t launch_nucleus_factory_list(const FactoryRequest* req32, const int* count32,
                                        int max32, const FactoryRequest* req64,
                                        const int* count64, int max64, cudaStream_t stream = 0);

}  // namespace g4gpu::npool

#endif
