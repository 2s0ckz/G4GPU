// The device tables P8b's hadronic wiring reads, built on the host and uploaded once per run.
//
// Three of the pieces `hadElastic` needs are TABLES that Geant4 builds in a
// `BuildPhysicsTable` call and then reads per step, and the rest is closed form. docs/RISK.md
// V5/V7 is why the distinction matters: Geant4 transports the table, not the model, so a port
// that evaluated the model per step would be right about the physics and wrong about the
// transport. What is uploaded here is what Geant4's own initialisation produces:
//
//   G4BGGNucleonElasticXS  per-Z Glauber-Gribov and Coulomb-barrier factors and theA[Z], for
//                          the proton. ~3 KB.
//   G4BGGPionElasticXS     the same for pi+ and pi-, plus G4UPiNuclearCrossSection's own
//                          tables, which are what the 20 MeV - 91 GeV band reads. ~30 KB.
//   G4ElasticHadrNucleusHE its `G4ElasticData`: 24 energies by up to 102 cumulative points per
//                          (hadron, Z), ~20 KB each, and it is the only one of the three that
//                          scales with the SCENE - built for pi+ and pi- and for the Z values
//                          the materials actually contain, because that is who reads it.
//
// WHY THE HE TABLE IS PER-SCENE AND THE OTHER TWO ARE NOT
//
// `G4ElasticHadrNucleusHE::SampleInvariantT` uses the table only for Z > 1 - a hadron-proton
// scatter is a closed-form `HadronProtonQ2` - and Geant4 itself builds a `G4ElasticData` lazily,
// the first time a (hadron, Z) pair is asked for. Building all 92 Z for both pions would be
// 3.7 MB and 184 numerical integrations of 24 energies each; building the scene's distinct Z is
// a handful. The BGG tables are per-Z arrays with no Z-dependent cost, so they are built whole.
//
// BUILT UNCONDITIONALLY, like the hadron range table beside it, and the cost is measured rather
// than guessed: 0.35 MB and 18 `G4ElasticData` for B1's nine distinct elements, each of which is
// 24 energies by up to 100 bins of a 10-point midpoint rule - about 430,000 evaluations of
// `he_hadr_nuc_differ_cr_sec` in total, under a second. `Upload` cannot know what species the
// generator will produce until it has produced one, so a gamma run pays this the way it already
// pays for the hadron range table. What a run with no charged hadron in it does NOT pay is any
// per-step cost: `elastic_xs_per_volume` returns zero for a species whose channel is `kNone`.
//
// A caller that wants the tables absent can leave the pointers null - `had::ElasticTables`'s
// default - and every channel that needs one then behaves exactly as a species with no elastic
// process does.
#pragma once

#include <cstdio>
#include <vector>

#include "data/atomic_masses.cuh"
#include "data/nuclei_mass_ame12.hh"
#include "physics/hadronic/elastic_wiring.cuh"

namespace g4gpu::host {

/// Everything `upload_elastic_tables` allocated, so a run can give it back.
template <typename real_t>
struct ElasticTableOwner {
  had::ElasticTables<real_t> view{};
  void* bgg_nucleon = nullptr;
  void* bgg_pion = nullptr;
  void* he = nullptr;
  void* he_slot = nullptr;
  void* he_grid = nullptr;
  void* he_bnd = nullptr;
  /// Device bytes the whole thing cost, for the upload report.
  std::size_t bytes = 0;
  int n_he_tables = 0;
};

/// Builds and uploads the elastic tables for a scene containing the elements @p zs.
///
/// @param zs the distinct atomic numbers in the scene, ascending. Only the entries above 1 get
///        an HE table; hydrogen is closed form.
/// @param verbose prints the memory cost, as `upload_photoelectric_for` and the range tables do.
template <typename real_t>
inline ElasticTableOwner<real_t> upload_elastic_tables(const std::vector<int>& zs,
                                                        bool verbose = true) {
  namespace el = g4gpu::physics::hadronic::elastic;
  namespace hxs = g4gpu::hadronic::xs;
  ElasticTableOwner<real_t> own;

  auto up = [&](const void* src, std::size_t n, void** out) {
    if (cudaMalloc(out, n) != cudaSuccess) {
      std::printf("\nFATAL: could not allocate %zu bytes for an elastic table\n", n);
      std::exit(1);
    }
    if (cudaMemcpy(*out, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of an elastic table\n", n);
      std::exit(1);
    }
    own.bytes += n;
  };

  // ---- G4BGGNucleonElasticXS, for the proton. QBBC builds the PROTON instance and gives the
  // neutron G4NeutronElasticXS instead, so `built_for_proton` is true - which is not decoration:
  // see bgg_build_nucleon_table's header for what a neutron instance built by a proton returns.
  {
    auto* h = new hxs::BggNucleonTable<real_t>();
    hxs::bgg_build_nucleon_table<real_t>(/*is_elastic=*/true, *h, /*built_for_proton=*/true);
    up(h, sizeof(*h), &own.bgg_nucleon);
    own.view.bgg_nucleon = static_cast<const hxs::BggNucleonTable<real_t>*>(own.bgg_nucleon);
    delete h;
  }

  // ---- G4BGGPionElasticXS, which carries both pion charges in one table.
  {
    auto* h = new hxs::BggPionTable<real_t>();
    hxs::bgg_build_pion_table<real_t>(/*is_elastic=*/true, *h);
    up(h, sizeof(*h), &own.bgg_pion);
    own.view.bgg_pion = static_cast<const hxs::BggPionTable<real_t>*>(own.bgg_pion);
    delete h;
  }

  // ---- G4ElasticHadrNucleusHE's energy grid and boundary constants.
  //
  // Uploaded rather than default-constructed on the device: `HeEnergyGrid`'s constructor
  // computes `exp(log(10)*0.1)` and twenty geometric steps from it, and a kernel that built one
  // per sampler call would pay twenty transcendentals for a table that is the same in every
  // step of every run.
  {
    auto* g = new el::HeEnergyGrid<real_t>();
    up(g, sizeof(*g), &own.he_grid);
    own.view.he_grid = static_cast<const el::HeEnergyGrid<real_t>*>(own.he_grid);
    delete g;
    auto* b = new el::HeBoundary<real_t>();
    up(b, sizeof(*b), &own.he_bnd);
    own.view.he_bnd = static_cast<const el::HeBoundary<real_t>*>(own.he_bnd);
    delete b;
  }

  // ---- the per-(pion, Z) G4ElasticData.
  {
    int max_z = 1;
    for (int z : zs) {
      if (z > max_z) { max_z = z; }
    }
    std::vector<short> slot_of_z(static_cast<std::size_t>(max_z) + 1, -1);
    std::vector<int> z_list;
    for (int z : zs) {
      if (z < 1 || z > max_z) { continue; }
      if (slot_of_z[z] >= 0) { continue; }
      slot_of_z[z] = static_cast<short>(z_list.size());
      z_list.push_back(z);
    }
    if (!z_list.empty()) {
      std::vector<el::HeElasticData<real_t>> tables(z_list.size() * had::kHeNumHadrons);
      const el::HeEnergyGrid<real_t> grid;
      const el::HeBoundary<real_t> bnd;
      const int* codes = el::he_hadron_code();
      const int* types = el::he_hadron_type();
      const int* types1 = el::he_hadron_type1();
      for (std::size_t i = 0; i < z_list.size(); ++i) {
        const int z = z_list[i];
        // `G4ElasticHadrNucleusHE::FillData` rounds the NIST atomic mass, which is
        // G4NistManager::GetAtomicMassAmu(Z) and not G4IsotopeList's aeff[Z].
        const int a = static_cast<int>(data::atomic_mass<real_t>(z) + real_t(0.5));
        const real_t mass_a = data::nuclear_mass<real_t>(a, z);
        for (int h = 0; h < had::kHeNumHadrons; ++h) {
          // h = 0 is pi+ and h = 1 is pi-, which are indices 0 and 1 of he_hadron_code() -
          // and that index is fElasticData's first subscript, so the two orders are one order.
          const real_t h_mass =
              particle_def<real_t>(h == 0 ? ParticleType::kPionPlus : ParticleType::kPionMinus)
                  .mass;
          tables[i * had::kHeNumHadrons + h] = el::he_fill_data<real_t>(
              z, a, mass_a, h_mass, types[h], types1[h], codes[h],
              /*is_proton_projectile=*/false, grid, bnd);
        }
      }
      up(tables.data(), tables.size() * sizeof(tables[0]), &own.he);
      own.view.he = static_cast<const el::HeElasticData<real_t>*>(own.he);
      up(slot_of_z.data(), slot_of_z.size() * sizeof(short), &own.he_slot);
      own.view.he_slot_of_z = static_cast<const short*>(own.he_slot);
      own.view.he_max_z = max_z;
      own.n_he_tables = static_cast<int>(tables.size());
    }
  }

  if (verbose) {
    std::printf("hadElastic tables: %.2f MB - BGG nucleon %zu B, BGG pion %zu B, "
                "%d G4ElasticData (%zu B each)\n",
                double(own.bytes) / 1048576.0, sizeof(hxs::BggNucleonTable<real_t>),
                sizeof(hxs::BggPionTable<real_t>), own.n_he_tables,
                sizeof(el::HeElasticData<real_t>));
  }
  return own;
}

template <typename real_t>
inline void free_elastic_tables(ElasticTableOwner<real_t>& own) {
  cudaFree(own.bgg_nucleon);
  cudaFree(own.bgg_pion);
  cudaFree(own.he);
  cudaFree(own.he_slot);
  cudaFree(own.he_grid);
  cudaFree(own.he_bnd);
  own = ElasticTableOwner<real_t>{};
}

}  // namespace g4gpu::host
