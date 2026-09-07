// The stock instantiation of the transport engine.
//
// The kernels and method bodies moved to transport_run_impl.cuh; read the note at the top of
// it before including it anywhere. This file exists to be the one place the default
// specialization - the null-gated StepTap that ships with the engine - is instantiated, so
// that every ordinary project links out/transport_run.obj and compiles no kernels of its own,
// exactly as before the split.
//
// A project with its own step hook does not touch this file. It includes the impl header in
// one .cu of its own and instantiates its own specialization there.
#include "host/transport_run_impl.cuh"

namespace g4gpu::host {

template class TransportEngine<double, StepTap<double>>;

}  // namespace g4gpu::host
