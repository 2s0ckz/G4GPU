// The neutron's cross sections on the device: the two per-process G4PARTICLEXS data sets and
// G4NeutronGeneralProcess's five combined tables.
//
// P8d. Everything here is an upload of something a HOST function already built and a test
// already compared with the oracle; no cross section is computed in this file.
//
// WHY THERE ARE TWO SETS OF TABLES AND NOT ONE, AND IT IS NOT A CHOICE
//
// docs/RISK.md V53 and its addendum: the port has no inelastic final state, and Geant4 cannot
// inactivate `neutronInelastic` while the general process holds it. So the like-for-like
// reference for a port without P9-P11 is a Geant4 whose `EnableNeutronGeneralProcess` is FALSE
// and whose `neutronInelastic` is then inactivated by name - at which point the neutron carries
// separate `hadElastic` and `nCapture` processes, each with its OWN cross-section data store and
// its own interaction length. That is a different competition from the general process's, not a
// component subtracted from it:
//
//   stage 1   lambda_el  = 1 / sum_i n_i sigma_el(Z_i, E)     G4NeutronElasticXS, per step
//             lambda_cap = 1 / sum_i n_i sigma_cap(Z_i, E)    G4NeutronCaptureXS, per step
//             two processes, two interaction lengths, smallest wins
//   final     lambda     = 1 / table0[material](E)            the general process's grid
//             the sub-process from table1/table2 (or table4 above 20 MeV)
//
// The two agree on nothing except the physics they are made of: the general process's table is a
// 401-node linear interpolation of the SUM evaluated at its own node energies, and the per-process
// path evaluates each data set at the track's energy. docs/PORTED.md 4.3's rule ("Geant4 does not
// run the model, it runs a table built from it") cuts both ways here - the per-process path is
// what Geant4 runs when the general process is off, because `G4HadronicProcess::GetMeanFreePath`
// asks the data store and the data store evaluates. So both are uploaded and the run-time
// `HadronicStage` picks.
//
// WHAT A `PxsDataSet` COSTS TO MOVE
//
// `data::ParticleXsTable` was written for this (see its own note about the three pointers): the
// slices, the isotope index and `max_z` are POD members, and the three `std::vector`s exist only
// so the host loader has somewhere to put the bytes. The device copy is the same struct with the
// vectors left EMPTY and `e_data`/`v_data`/`iso_data` pointing at device allocations - which is
// the shape that note predicted, verbatim. Nothing on the device touches a vector.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "data/materials.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/neutron_general_xs.cuh"
#include "physics/hadronic/neutron_wiring.cuh"
#include "physics/hadronic/xs/neutron_general_xs.cuh"
#include "physics/hadronic/xs/particlexs.cuh"

namespace g4gpu::host {

/// Every device allocation the neutron's physics needs, so a run can give them back.
template <typename real_t>
struct NeutronTableOwner {
  /// The two per-process data sets - `HadronicStage::kStage1` reads these.
  had::NeutronSubTables<real_t> sub{};
  /// The combined table - `HadronicStage::kFinal` reads this.
  had::NeutronGeneralXs<real_t>* d_general = nullptr;

  std::vector<void*> allocs;
  std::size_t bytes = 0;
  int n_low = 0, n_high = 0, n_mat = 0;
  /// False when G4PARTICLEXSDATA could not be resolved. Nothing is uploaded then and the
  /// neutron has no hadronic process, which is the state before this package.
  bool ok = false;
};

namespace detail {

/// One `PxsDataSet` and the table under it, on the device.
///
/// Four allocations: the concatenated energies, the concatenated values, the isotope slices, and
/// then the two structs. The order matters only in that the structs are uploaded LAST, after
/// their pointers have been repointed.
template <typename real_t>
inline const hadronic::xs::PxsDataSet<real_t>* upload_pxs(
    const data::ParticleXsTable<real_t>& h_table,
    const hadronic::xs::PxsDataSet<real_t>& h_ds, NeutronTableOwner<real_t>& own) {
  auto up = [&](const void* src, std::size_t n) -> void* {
    if (n == 0) { return nullptr; }
    void* p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess) {
      std::printf("\nFATAL: could not allocate %zu bytes for a neutron cross-section table\n", n);
      std::exit(1);
    }
    if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of a neutron cross-section table\n", n);
      std::exit(1);
    }
    own.allocs.push_back(p);
    own.bytes += n;
    return p;
  };

  void* d_e = up(h_table.table_e.data(), h_table.table_e.size() * sizeof(real_t));
  void* d_v = up(h_table.table_v.data(), h_table.table_v.size() * sizeof(real_t));
  void* d_iso = up(h_table.isotopes.data(),
                   h_table.isotopes.size() * sizeof(data::PxsSlice<real_t>));

  // A HOST STRUCT WITH DEVICE POINTERS AND NO VECTORS, built with `new` rather than on the
  // stack: `ParticleXsTable<double>` is about 4 KB of slices and this runs during Upload, not
  // in a kernel. The three vectors are left default-constructed - an empty std::vector is a
  // handful of null pointers, and the device never reads them, which is the arrangement
  // data/particlexs_data.cuh's own comment describes ("A device build replaces these three with
  // device allocations and leaves the vectors empty").
  auto* stage = new data::ParticleXsTable<real_t>();
  for (int z = 0; z < data::kPxsMaxZ; ++z) {
    stage->element[z] = h_table.element[z];
    stage->iso_base[z] = h_table.iso_base[z];
    stage->iso_n[z] = h_table.iso_n[z];
  }
  stage->e_data = static_cast<const real_t*>(d_e);
  stage->v_data = static_cast<const real_t*>(d_v);
  stage->iso_data = static_cast<const data::PxsSlice<real_t>*>(d_iso);
  stage->max_z = h_table.max_z;
  void* d_table = up(stage, sizeof(*stage));
  delete stage;

  auto* ds = new hadronic::xs::PxsDataSet<real_t>(h_ds);
  ds->data = static_cast<const data::ParticleXsTable<real_t>*>(d_table);
  void* d_ds = up(ds, sizeof(*ds));
  delete ds;
  return static_cast<const hadronic::xs::PxsDataSet<real_t>*>(d_ds);
}

}  // namespace detail

/// Reads `G4PARTICLEXS4.0/neutron/{el,cap}`, builds G4NeutronGeneralProcess's five tables for
/// the scene's materials, and uploads all of it.
///
/// @param mats the scene's materials, host-side - `ngp_build_table` needs the atom densities to
///        make a MACROSCOPIC cross section, so the combined table is per scene and not per
///        element like the BGG ones.
/// @param verbose prints the memory cost, as every other uploader in this directory does.
///
/// THE INELASTIC DATA SET IS READ AND ITS FINAL STATE IS NOT WRITTEN, and both halves of that
/// are deliberate. `G4NeutronGeneralProcess::BuildPhysicsTable` sums elastic + inelastic +
/// capture UNCONDITIONALLY (docs/RISK.md V53), so table 0 is wrong without the inelastic term
/// and the final configuration's interaction length would be too long without it. What the port
/// cannot do is APPLY the inelastic final state, and that is refused by name where it would be
/// needed - `had::HadronicRefusal::kNeutronInelastic` - rather than by leaving a term out of a
/// cross section and calling the result the general process.
template <typename real_t>
inline NeutronTableOwner<real_t> upload_neutron_tables(const data::Material<real_t>* mats,
                                                        int n_mat, bool verbose = true) {
  namespace hxs = g4gpu::hadronic::xs;
  NeutronTableOwner<real_t> own;

  const std::string d_el = g4particlexs_subdir("neutron");
  if (d_el.empty()) {
    if (verbose) {
      std::printf("neutron tables: G4PARTICLEXSDATA could not be resolved - the neutron has no "
                  "hadronic process\n");
    }
    return own;
  }

  // The three data sets G4NeutronGeneralProcess takes as `InitialisationXS(proc)`, which is the
  // FIRST data set of each sub-process's store: G4NeutronElasticXS, G4NeutronInelasticXS and
  // G4NeutronCaptureXS. Held in `new`ed storage because a `PxsDataSet` points into its table
  // and both have to outlive the build.
  auto* t_el = new data::ParticleXsTable<real_t>();
  auto* t_inel = new data::ParticleXsTable<real_t>();
  auto* t_cap = new data::ParticleXsTable<real_t>();
  hxs::PxsDataSet<real_t> ds_el, ds_inel, ds_cap;
  const bool got =
      hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronElastic, hxs::neutron<real_t>(), d_el, *t_el,
                            ds_el)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronInelastic, hxs::neutron<real_t>(), d_el,
                               *t_inel, ds_inel)
      && hxs::pxs_load<real_t>(hxs::PxsKind::kNeutronCapture, hxs::neutron<real_t>(), d_el,
                               *t_cap, ds_cap);
  if (!got) {
    std::printf("\nFATAL: G4PARTICLEXS4.0/neutron is incomplete - an element file is missing.\n"
                "  A missing dataset is fatal here as everywhere (src/host/g4data.cuh), because\n"
                "  a neutron with a partial cross section is a transport that is quietly wrong\n"
                "  in one element.\n");
    std::exit(1);
  }

  own.sub.elastic = detail::upload_pxs<real_t>(*t_el, ds_el, own);
  own.sub.capture = detail::upload_pxs<real_t>(*t_cap, ds_cap, own);

  // ---- G4NeutronGeneralProcess's five tables, for this scene's materials.
  auto* gt = new hxs::NeutronGeneralTable<real_t>();
  hxs::ngp_build_table<real_t>(ds_el, ds_inel, ds_cap, mats, n_mat, *gt);
  own.n_low = gt->n_low;
  own.n_high = gt->n_high;
  own.n_mat = gt->n_mat;

  auto up = [&](const void* src, std::size_t n) -> void* {
    if (n == 0) { return nullptr; }
    void* p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess) {
      std::printf("\nFATAL: could not allocate %zu bytes for the neutron general table\n", n);
      std::exit(1);
    }
    if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of the neutron general table\n", n);
      std::exit(1);
    }
    own.allocs.push_back(p);
    own.bytes += n;
    return p;
  };

  // The two grids and the five value arrays, then five arrays of PhysVec whose `e` and `v` are
  // repointed at them. The PhysVec's own `Initialise()` results - edge_min, edge_max, inv_dbin,
  // log_emin - travel with it, so the device recomputes no logarithm per lookup and the node
  // energies are the ones the table was BUILT on. docs/RISK.md V53 is what the alternative cost.
  const real_t* d_elow = static_cast<const real_t*>(
      up(gt->e_low.data(), gt->e_low.size() * sizeof(real_t)));
  const real_t* d_ehigh = static_cast<const real_t*>(
      up(gt->e_high.data(), gt->e_high.size() * sizeof(real_t)));

  auto up_rows = [&](const std::vector<real_t>& vals, const real_t* d_grid, int n,
                      const std::vector<hxs::PhysVec<real_t>>& h_views)
      -> const hxs::PhysVec<real_t>* {
    const real_t* d_vals = static_cast<const real_t*>(up(vals.data(),
                                                         vals.size() * sizeof(real_t)));
    std::vector<hxs::PhysVec<real_t>> v = h_views;
    for (int m = 0; m < n_mat; ++m) {
      v[static_cast<std::size_t>(m)].e = d_grid;
      v[static_cast<std::size_t>(m)].v = d_vals + static_cast<std::size_t>(m) * n;
    }
    return static_cast<const hxs::PhysVec<real_t>*>(
        up(v.data(), v.size() * sizeof(hxs::PhysVec<real_t>)));
  };

  had::NeutronGeneralXs<real_t> view;
  view.t0 = up_rows(gt->t0, d_elow, gt->n_low, gt->v0);
  view.t1 = up_rows(gt->t1, d_elow, gt->n_low, gt->v1);
  view.t2 = up_rows(gt->t2, d_elow, gt->n_low, gt->v2);
  view.t3 = up_rows(gt->t3, d_ehigh, gt->n_high, gt->v3);
  view.t4 = up_rows(gt->t4, d_ehigh, gt->n_high, gt->v4);
  view.n_materials = gt->n_mat;
  own.d_general = static_cast<had::NeutronGeneralXs<real_t>*>(up(&view, sizeof(view)));

  if (verbose) {
    std::printf("neutron tables: %.3f MB - G4NeutronElasticXS and G4NeutronCaptureXS on the "
                "device, G4NeutronGeneralProcess %d+%d nodes x %d materials\n",
                double(own.bytes) / 1048576.0, gt->n_low, gt->n_high, gt->n_mat);
  }
  delete gt;
  delete t_el;
  delete t_inel;
  delete t_cap;
  own.ok = true;
  return own;
}

template <typename real_t>
inline void free_neutron_tables(NeutronTableOwner<real_t>& own) {
  for (void* p : own.allocs) { cudaFree(p); }
  own = NeutronTableOwner<real_t>{};
}

}  // namespace g4gpu::host
