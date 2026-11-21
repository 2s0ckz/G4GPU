// P38: the nucleus factory's unit (docs/RISK.md V319) - the four block widths of
// `npool::nucleus_factory_kernel` and their launch, and nothing else of the engine.
//
// A unit of its own because the factory is a family of small kernels (72-80 registers, a
// ~230-byte frame, ~10 s of ptxas) that must not land in an interaction unit's 27 GB ptxas, and
// because the engine's own unit carries no kernel (build_engine.bat's check). P39 (V321) made
// the flat cascade's set-up its first caller in the transport: the two request-list widths.
#include <algorithm>

#include "physics/hadronic/npool/nucleus_factory.cuh"
#include "physics/hadronic/npool/nucleus_factory_launch.cuh"

namespace g4gpu::npool {

cudaError_t launch_nucleus_factory(const FactoryJob& job, const FactoryOutput& out,
                                   cudaStream_t stream, int grid) {
  if (job.n <= 0) { return cudaSuccess; }
  if (job.a < 1 || job.a > kFactoryMaxA) { return cudaErrorInvalidValue; }
  const int g = (grid > 0) ? grid : std::min(job.n, 65535);
  switch (factory_width(job.a)) {
    case 32: nucleus_factory_kernel<32><<<g, 32, 0, stream>>>(job, out); break;
    case 64: nucleus_factory_kernel<64><<<g, 64, 0, stream>>>(job, out); break;
    case 128: nucleus_factory_kernel<128><<<g, 128, 0, stream>>>(job, out); break;
    default: nucleus_factory_kernel<256><<<g, 256, 0, stream>>>(job, out); break;
  }
  return cudaGetLastError();
}

cudaError_t launch_nucleus_factory_list(const FactoryRequest* req32, const int* count32,
                                        int max32, const FactoryRequest* req64,
                                        const int* count64, int max64, cudaStream_t stream) {
  // A block a request up to 65,535 blocks, striding past that: a block past the device count
  // returns at once, so an upper bound costs a launch of empty blocks and nothing else.
  if (max32 > 0) {
    nucleus_factory_list_kernel<32><<<std::min(max32, 65535), 32, 0, stream>>>(req32, count32, max32);
  }
  if (max64 > 0) {
    nucleus_factory_list_kernel<64><<<std::min(max64, 65535), 64, 0, stream>>>(req64, count64, max64);
  }
  return cudaGetLastError();
}

}  // namespace g4gpu::npool
