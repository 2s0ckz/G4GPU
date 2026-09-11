// The stock instantiation of the transport engine's HOST code.
//
// The kernels and method bodies moved to transport_run_impl.cuh; read the note at the top of
// it before including it anywhere. This file exists to be the one place the default
// specialization - the null-gated StepTap that ships with the engine - is instantiated, so
// that every ordinary project links out/transport_run.lib and compiles no kernels of its own,
// exactly as before the split.
//
// WHAT THIS FILE DOES *NOT* COMPILE ANY MORE, AND WHY THAT IS THE POINT
//
// Until P8e this was also the only place the eighteen stepping kernels were instantiated -
// implicitly, by the launches in BeamOn - and the file took about 24 minutes of nvcc. With the
// ion's Urban multiple scattering dispatched in step_hadron it stopped compiling at all
// (docs/RISK.md V63: `ptxas died with status 0xC0000005`, deterministically, in nine different
// arrangements of the same code). The explicit instantiation declarations under the kernels in
// transport_run_impl.cuh now keep every one of them out of this translation unit, which
// therefore holds the engine's host code plus the three utility kernels that carry no physics -
// seed_from_primaries, count_species and scatter_species. It compiles in 68 seconds. The
// stepping kernels are sixteen translation units beside this one, one per kernel, which
// build_engine.bat compiles six at a time and archives into out/transport_run.lib.
// docs/RISK.md V65.
//
// A project with its own step hook does not touch this file. It includes the impl header in
// one .cu of its own and instantiates its own specialization there.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template class TransportEngine<double, StepTap<double>>;

}  // namespace g4gpu::host
