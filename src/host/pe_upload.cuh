// Loads the Livermore photoelectric data and uploads it to the device.
//
// The table struct holds pointers into two shared arrays, so those are uploaded first and
// the struct is patched to point at device memory before it is itself copied over.
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>
#include "data/brems_data.cuh"
#include "data/photoelectric_data.cuh"
#include "data/rayleigh_data.cuh"
#include "host/g4data.cuh"
#include "physics/em/electron_processes.cuh"
#include "physics/em/urban_msc.cuh"
#include "physics/em/wentzel_msc.cuh"

namespace g4gpu::host {

/// A missing dataset is fatal, not a warning. Running without photoelectric or Rayleigh
/// still produces a plausible-looking dose - about half a percent low for B1 - which is far
/// worse than not running at all, because nothing downstream flags it. This is how a shell
/// with G4LEDATA pointing at an old G4EMLOW quietly turned two processes off.
[[noreturn]] inline void fatal_missing_data(const char* what, const std::string& dir) {
  std::printf("\nFATAL: %s data not found.\n  looked in: %s\n", what,
              dir.empty() ? "(no G4EMLOW directory located)" : dir.c_str());
  std::printf("  Set G4LEDATA to a G4EMLOW 8.x directory, or GEANT4_DATA_DIR to the\n"
              "  share/Geant4/data folder that contains it.\n");
  std::exit(2);
}

[[noreturn]] inline void fatal_upload(const char* what) {
  std::printf("\nFATAL: could not upload %s tables to the device.\n", what);
  std::exit(2);
}

/// What the e+- energy-loss tables cost, reported like every other table's line.
///
/// Worth a line of its own because the object grew by a factor of eight when the ceiling came
/// off: it was one species on 128 bins to 100 MeV and is now two species on Geant4's 85-node
/// grid to 100 TeV with five arrays each (dE/dx, range, and three sets of spline second
/// derivatives). docs/RISK.md V64.
template <typename real_t>
inline void print_electron_table_cost(int n_materials) {
  std::printf("e+- tables: dE/dx, range and inverse range for e- and e+, %d materials,\n"
              "  %d bins (100 eV to 100 TeV, 7 per decade, cubic spline), %.1f KiB\n",
              n_materials, em::kRangeBins, sizeof(em::RangeTable<real_t>) / 1024.0);
}

/// EPICS2017 photoelectric cross sections, inside whichever G4EMLOW g4data.cuh located.
/// G4GPU_PHOT_DIR overrides it outright, for pointing at a directory of test data.
inline std::string default_phot_dir() {
  if (const char* env = std::getenv("G4GPU_PHOT_DIR")) { return env; }
  return g4emlow_subdir("epics2017/phot", "pe-cs-1.dat");
}

/// Every element appearing in B1's four materials.
inline const int* b1_elements(int& n) {
  static const int zs[10] = {1, 6, 7, 8, 9, 12, 15, 16, 18, 20};
  n = 10;
  return zs;
}

#ifdef __CUDACC__
/// Returns a device pointer to the uploaded table, or nullptr if the data could not be read.
/// Failure is loud: a silently absent photoelectric process would just look like a small
/// dose deficit later.
template <typename real_t>
inline data::PhotoElectricTable<real_t>* upload_photoelectric(const std::string& dir) {
  int n_z = 0;
  const int* zs = b1_elements(n_z);

  data::PhotoElectricTable<real_t> host_table{};
  std::vector<real_t> te, tv;
  if (!data::load_photoelectric<real_t>(dir, zs, n_z, host_table, te, tv)) {
    fatal_missing_data("photoelectric (G4EMLOW epics2017/phot)", dir);
    return nullptr;
  }

  real_t* d_te = nullptr;
  real_t* d_tv = nullptr;
  data::PhotoElectricTable<real_t>* d_table = nullptr;
  const size_t nb = sizeof(real_t) * te.size();
  if (cudaMalloc(&d_te, nb) != cudaSuccess || cudaMalloc(&d_tv, nb) != cudaSuccess
      || cudaMalloc(&d_table, sizeof(host_table)) != cudaSuccess) {
    fatal_upload("photoelectric");
    return nullptr;
  }
  cudaMemcpy(d_te, te.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tv, tv.data(), nb, cudaMemcpyHostToDevice);

  host_table.table_e = d_te;
  host_table.table_v = d_tv;
  host_table.table_n = static_cast<int>(te.size());
  cudaMemcpy(d_table, &host_table, sizeof(host_table), cudaMemcpyHostToDevice);

  std::printf("photoelectric: %d elements, %d tabulated points\n", host_table.n_elements,
              host_table.table_n);
  return d_table;
}
#endif
/// EPICS2017 Rayleigh cross sections and atomic form factors.
inline std::string default_rayl_dir() {
  if (const char* env = std::getenv("G4GPU_RAYL_DIR")) { return env; }
  return g4emlow_subdir("epics2017/rayl", "re-cs-1.dat");
}

/// Seltzer-Berger scaled bremsstrahlung differential cross sections.
inline std::string default_sb_dir() {
  if (const char* env = std::getenv("G4GPU_SB_DIR")) { return env; }
  return g4emlow_subdir("brem_SB", "br1");
}

#ifdef __CUDACC__
template <typename real_t>
struct BremsUpload {
  data::BremsTable<real_t>* table = nullptr;
  data::SBTableSet<real_t>* sb = nullptr;
};

/// Loads the Seltzer-Berger DCS tables, integrates them into per-material dE/dx and
/// cross-section tables on the host, uploads both, and **rebuilds the range table** so it
/// integrates the real restricted radiative stopping power instead of the yield scaling.
template <typename real_t>
inline BremsUpload<real_t> upload_brems(const std::string& dir,
                                        const data::Material<real_t>* h_mats,
                                        em::RangeTable<real_t>& h_rt) {
  BremsUpload<real_t> out;
  int n_z = 0;
  const int* zs = b1_elements(n_z);

  static data::SBTableSet<real_t> h_sb{};
  if (!data::load_sb_tables<real_t>(dir, zs, n_z, h_sb)) {
    fatal_missing_data("Seltzer-Berger bremsstrahlung (G4EMLOW brem_SB)", dir);
    return out;
  }
  static data::BremsTable<real_t> h_bt;
  data::build_brems_tables<real_t>(h_mats, h_sb, h_bt);

  // The e+- dE/dx table is the sum over G4eIonisation AND G4eBremsstrahlung, so it can only
  // be built once the Seltzer-Berger tables are loaded - which is why it is built here and
  // not beside the materials. It takes the SB tables and not `h_bt`: see
  // `em::brems_restricted_dedx`, which evaluates the models at this table's own 85 nodes
  // because 6 of every 7 of them fall between `h_bt`'s.
  em::build_range_table<real_t>(h_mats, h_rt, &h_sb);
  print_electron_table_cost<real_t>(data::kNumMaterials);

  if (cudaMalloc(&out.table, sizeof(h_bt)) != cudaSuccess
      || cudaMalloc(&out.sb, sizeof(h_sb)) != cudaSuccess) {
    fatal_upload("bremsstrahlung");
    out.table = nullptr;
    out.sb = nullptr;
    return out;
  }
  cudaMemcpy(out.table, &h_bt, sizeof(h_bt), cudaMemcpyHostToDevice);
  cudaMemcpy(out.sb, &h_sb, sizeof(h_sb), cudaMemcpyHostToDevice);
  std::printf("bremsstrahlung: %d Seltzer-Berger element tables, %d energy bins\n",
              h_sb.n_elements, data::kBremsBins);
  return out;
}
#endif

#ifdef __CUDACC__
/// Uploads the Livermore Rayleigh tables; nullptr disables coherent scattering.
template <typename real_t>
inline data::RayleighTable<real_t>* upload_rayleigh(const std::string& dir, const int* zs,
                                                    int n_z) {
  data::RayleighTable<real_t> h{};
  std::vector<real_t> te, tv;
  if (!data::load_rayleigh<real_t>(dir, zs, n_z, h, te, tv)) {
    fatal_missing_data("Rayleigh (G4EMLOW epics2017/rayl)", dir);
    return nullptr;
  }
  real_t* d_te = nullptr;
  real_t* d_tv = nullptr;
  data::RayleighTable<real_t>* d_t = nullptr;
  const size_t nb = sizeof(real_t) * te.size();
  if (cudaMalloc(&d_te, nb) != cudaSuccess || cudaMalloc(&d_tv, nb) != cudaSuccess
      || cudaMalloc(&d_t, sizeof(h)) != cudaSuccess) {
    fatal_upload("Rayleigh");
    return nullptr;
  }
  cudaMemcpy(d_te, te.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tv, tv.data(), nb, cudaMemcpyHostToDevice);
  h.table_e = d_te;
  h.table_v = d_tv;
  h.table_n = static_cast<int>(te.size());
  cudaMemcpy(d_t, &h, sizeof(h), cudaMemcpyHostToDevice);
  std::printf("rayleigh: %d elements, %d tabulated points\n", h.n_elements, h.table_n);
  return d_t;
}
#endif

/// Builds the Urban MSC table on the host and uploads it. Depends only on the material
/// table, so it works for any user-supplied materials.
template <typename real_t>
inline em::UrbanTable<real_t>* upload_msc(const data::Material<real_t>* mats, int n_materials) {
  auto* h = new em::UrbanTable<real_t>();
  em::build_urban_table<real_t>(mats, n_materials, *h);
  em::UrbanTable<real_t>* d = nullptr;
  cudaMalloc(&d, sizeof(em::UrbanTable<real_t>));
  cudaMemcpy(d, h, sizeof(em::UrbanTable<real_t>), cudaMemcpyHostToDevice);
  std::printf("msc: Urban transport mfp for %d materials, %d energy bins, %.1f KiB\n",
              n_materials, em::kMscBins, sizeof(em::UrbanTable<real_t>) / 1024.0);
  delete h;
  return d;
}

/// Builds and uploads `G4VMscModel::xSectionTable` for the e+- WentzelVI model - the transport
/// mean free path from `G4EmParameters::MscEnergyLimit()` to `MaxKinEnergy`.
///
/// A second msc table and not an extension of the first, because Geant4 has two: one per
/// MODEL, each over that model's own energy window (`G4LossTableBuilder::BuildTableForModel`).
/// `em/wentzel_msc.cuh`'s header block has the grid, the stored quantity and the cut.
template <typename real_t>
inline em::WentzelLeptonTable<real_t>* upload_wv_lepton(const data::Material<real_t>* mats,
                                                        int n_materials) {
  auto* h = new em::WentzelLeptonTable<real_t>();
  em::build_wentzel_lepton_table<real_t>(mats, n_materials, *h);
  em::WentzelLeptonTable<real_t>* d = nullptr;
  if (cudaMalloc(&d, sizeof(em::WentzelLeptonTable<real_t>)) != cudaSuccess) {
    delete h;
    fatal_upload("e+- WentzelVI transport mfp");
  }
  cudaMemcpy(d, h, sizeof(em::WentzelLeptonTable<real_t>), cudaMemcpyHostToDevice);
  std::printf("msc: WentzelVI transport mfp for e+- above %g MeV, %d materials, %d bins, "
              "%.1f KiB\n",
              em::kWvLeptonEMin, n_materials, em::kWvLeptonBins,
              sizeof(em::WentzelLeptonTable<real_t>) / 1024.0);
  delete h;
  return d;
}

#ifdef __CUDACC__
// ------------------------------------------------- scene-driven variants
//
// The B1 helpers above take their element list from b1_elements(); these take whatever
// elements the scene's materials actually contain, and whatever number of materials there
// are. That is the difference between running example B1 and running an arbitrary detector.

template <typename real_t>
inline data::PhotoElectricTable<real_t>* upload_photoelectric_for(const std::string& dir,
                                                                  const std::vector<int>& zs) {
  data::PhotoElectricTable<real_t> host_table{};
  std::vector<real_t> te, tv;
  if (!data::load_photoelectric<real_t>(dir, zs.data(), static_cast<int>(zs.size()), host_table,
                                        te, tv)) {
    fatal_missing_data("photoelectric (G4EMLOW epics2017/phot)", dir);
    return nullptr;
  }
  real_t* d_te = nullptr;
  real_t* d_tv = nullptr;
  data::PhotoElectricTable<real_t>* d_table = nullptr;
  const size_t nb = sizeof(real_t) * te.size();
  if (cudaMalloc(&d_te, nb) != cudaSuccess || cudaMalloc(&d_tv, nb) != cudaSuccess
      || cudaMalloc(&d_table, sizeof(host_table)) != cudaSuccess) {
    fatal_upload("photoelectric");
    return nullptr;
  }
  cudaMemcpy(d_te, te.data(), nb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_tv, tv.data(), nb, cudaMemcpyHostToDevice);
  host_table.table_e = d_te;
  host_table.table_v = d_tv;
  host_table.table_n = static_cast<int>(te.size());
  cudaMemcpy(d_table, &host_table, sizeof(host_table), cudaMemcpyHostToDevice);
  std::printf("photoelectric: %d elements, %d tabulated points\n", host_table.n_elements,
              host_table.table_n);
  return d_table;
}

template <typename real_t>
inline BremsUpload<real_t> upload_brems_for(const std::string& dir,
                                            const data::Material<real_t>* h_mats,
                                            int n_materials, em::RangeTable<real_t>& h_rt,
                                            const std::vector<int>& zs) {
  BremsUpload<real_t> out;
  static data::SBTableSet<real_t> h_sb{};
  if (!data::load_sb_tables<real_t>(dir, zs.data(), static_cast<int>(zs.size()), h_sb)) {
    fatal_missing_data("Seltzer-Berger bremsstrahlung (G4EMLOW brem_SB)", dir);
    return out;
  }
  static data::BremsTable<real_t> h_bt;
  data::build_brems_tables<real_t>(h_mats, h_sb, h_bt, n_materials);
  // See the note in `upload_brems`: the e+- dE/dx table needs the radiative term, and it
  // takes the SB tables rather than `h_bt` so that it samples the models on its own grid.
  em::build_range_table<real_t>(h_mats, h_rt, &h_sb, n_materials);
  print_electron_table_cost<real_t>(n_materials);

  if (cudaMalloc(&out.table, sizeof(h_bt)) != cudaSuccess
      || cudaMalloc(&out.sb, sizeof(h_sb)) != cudaSuccess) {
    fatal_upload("bremsstrahlung");
    out.table = nullptr;
    out.sb = nullptr;
    return out;
  }
  cudaMemcpy(out.table, &h_bt, sizeof(h_bt), cudaMemcpyHostToDevice);
  cudaMemcpy(out.sb, &h_sb, sizeof(h_sb), cudaMemcpyHostToDevice);
  std::printf("bremsstrahlung: %d Seltzer-Berger element tables, %d energy bins\n",
              h_sb.n_elements, data::kBremsBins);
  return out;
}
#endif

}  // namespace g4gpu::host
