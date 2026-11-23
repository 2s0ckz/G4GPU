// P44 (docs/RISK.md V351): builds the e+- lambda tables of `em/lepton_lambda.cuh` for the scene's
// materials on the host and uploads them - the device half of Geant4's integral approach for
// eIoni, eBrem and the annihilation.
//
// A FILE OF ITS OWN rather than a fourth table inside `upload_brems_for`, whose Seltzer-Berger set
// is a function-local static: the brems model needs those tables to tabulate its cross section on
// Geant4's lambda grid, so they are loaded here again (the same files, the scene's elements - a
// few megabytes read once at upload). `G4GPU_EM_INTEGRAL=0` skips the build and returns null,
// which `Scene::lepton_lambda` reads as the pre-P44 arithmetic.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "data/brems_data.cuh"
#include "data/materials.cuh"
#include "physics/em/lepton_lambda.cuh"

namespace g4gpu::host {

/// True unless the run asked for the models at the pre-step energy (`G4GPU_EM_INTEGRAL=0`).
inline bool lepton_integral_requested() {
  const char* env = std::getenv("G4GPU_EM_INTEGRAL");
  return !(env != nullptr && env[0] == '0');
}

#ifdef __CUDACC__
/// Null with integral off, or when the build or the upload fails - the latter loudly.
template <typename real_t>
inline em::LeptonLambdaTables<real_t>* upload_lepton_lambda(const std::string& sb_dir,
                                                            const data::Material<real_t>* h_mats,
                                                            int n_materials,
                                                            const std::vector<int>& zs) {
  if (!lepton_integral_requested()) {
    std::printf("e+- integral approach: OFF (G4GPU_EM_INTEGRAL=0) - the models at the pre-step "
                "energy, no rejection\n");
    return nullptr;
  }
  static data::SBTableSet<real_t> h_sb{};
  if (!data::load_sb_tables<real_t>(sb_dir, zs.data(), static_cast<int>(zs.size()), h_sb)) {
    std::printf("FATAL: e+- lambda tables: no Seltzer-Berger tables in %s\n", sb_dir.c_str());
    std::exit(1);
  }
  static em::LeptonLambdaTables<real_t> h{};
  if (!em::build_lepton_lambda_tables<real_t>(h_mats, n_materials, h_sb, h)) {
    std::printf("FATAL: e+- lambda tables could not be built\n");
    std::exit(1);
  }
  em::LeptonLambdaTables<real_t>* d = nullptr;
  if (cudaMalloc(&d, sizeof(h)) != cudaSuccess
      || cudaMemcpy(d, &h, sizeof(h), cudaMemcpyHostToDevice) != cudaSuccess) {
    std::printf("FATAL: e+- lambda tables: device upload of %zu bytes failed\n", sizeof(h));
    std::exit(1);
  }
  std::printf("e+- integral approach: eIoni and eBrem lambda tables for %d materials (%zu kB), "
              "e- eIoni type %d, eBrem type %d\n",
              n_materials, sizeof(h) / 1024, static_cast<int>(h.ioni_type[0]),
              static_cast<int>(h.brem_type[0]));
  return d;
}
#endif

}  // namespace g4gpu::host
