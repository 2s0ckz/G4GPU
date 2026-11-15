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
#include <string>
#include <vector>

#include "data/atomic_masses.cuh"
#include "data/nuclei_mass_ame12.hh"
#include "host/g4data.cuh"
#include "host/neutron_upload.cuh"
#include "physics/hadronic/elastic_wiring.cuh"
// P19: the muon model's sampling table, its constants and its host builder - and not the model,
// whose header reaches Bertini and the gamma chain. Until docs/RISK.md V233 this included
// `lepton_vd.cuh`, and every translation unit that uploads a scene read them; P3's Fermi
// break-up pool's upload left for host/interaction_pool_build.cuh for the same reason.
#include "physics/hadronic/emextra/mu_vd_table.cuh"
#include "physics/hadronic/emextra_wiring.cuh"
#include "physics/hadronic/inelastic_wiring.cuh"
// P26: G4VEmProcess's lambda tables for the photon's four processes, host only, which the general
// process's tables are built from.
#include "physics/em/em_lambda_table.cuh"
#include "physics/em/pair_production.cuh"

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

// =============================================================================================
// P15: the INELASTIC cross sections
// =============================================================================================
//
// Five `G4ParticleInelasticXS` data sets and one `G4BGGPionInelasticXS` table. The neutron's
// `G4NeutronInelasticXS` is NOT here - it is loaded and uploaded beside the other two neutron
// data sets in `host/neutron_upload.cuh`, because all three have to come from the same load
// that builds `G4NeutronGeneralProcess`'s combined table or the sub-process partials and the
// total would be two different numbers.
//
// FIVE DATA SETS AND NOT ONE SCALED. `G4ParticleInelasticXS`'s constructor takes the particle
// and reads `G4PARTICLEXS4.0/<name>/inel<Z>` for it - `proton/`, `deuteron/`, `triton/`,
// `He3/`, `alpha/` are five directories of per-element and per-isotope files - and it picks the
// high-energy hand-over component by particle name as well: Glauber-Gribov hadron-nucleus for
// the proton and Glauber-Gribov NUCL-NUCL for the four ions (`pxs_high_energy`'s switch). A
// port that loaded `proton/` once and scaled by A would be wrong in both halves.
//
// WHAT IT COSTS, measured by the printout below rather than estimated: the five tables together
// are about 3 MB of device memory for a full 92-element load, against the 9.52 MB of
// PhotonEvaporation level data already uploaded unconditionally. Built here beside the elastic
// tables and for the same reason: a run that will carry a charged hadron needs them before the
// first primary is seeded.

/// Everything `upload_inelastic_tables` allocated, so a run can give it back.
template <typename real_t>
struct InelasticTableOwner {
  had::InelasticTables<real_t> view{};
  std::vector<void*> allocs;
  std::size_t bytes = 0;
  /// False when `G4PARTICLEXSDATA` could not be resolved. Nothing is uploaded then and every
  /// species' inelastic channel behaves exactly as a species with no such process does - an
  /// infinite interaction length and no uniform drawn.
  bool ok = false;
  int n_pxs = 0;
};

/// Loads and uploads the five `G4ParticleInelasticXS` data sets and `G4BGGPionInelasticXS`.
///
/// A missing dataset is FATAL, here as everywhere (`src/host/g4data.cuh`), for the reason that
/// file states: a hadron with a partial cross section is a transport that is quietly wrong in
/// one element. The one thing that is NOT fatal is the whole `G4PARTICLEXSDATA` variable being
/// unset, which is the pre-P2 state and leaves every inelastic channel switched off.
template <typename real_t>
inline InelasticTableOwner<real_t> upload_inelastic_tables(bool verbose = true) {
  namespace hxs = g4gpu::hadronic::xs;
  InelasticTableOwner<real_t> own;

  // The five species, in `had::particle_inelastic_slot` order. The directory names are
  // `G4ParticleDefinition::GetParticleName()`, which is what `G4ParticleInelasticXS`'s
  // constructor builds its path from.
  struct Row {
    const char* dir;
    hxs::Projectile<real_t> (*proj)();
  };
  const Row rows[5] = {
      {"proton",   &hxs::proton<real_t>},
      {"deuteron", &hxs::deuteron<real_t>},
      {"triton",   &hxs::triton<real_t>},
      {"He3",      &hxs::he3<real_t>},
      {"alpha",    &hxs::alpha<real_t>},
  };

  if (g4particlexs_subdir("proton").empty()) {
    if (verbose) {
      std::printf("inelastic tables: G4PARTICLEXSDATA could not be resolved - no charged "
                  "hadron has an inelastic process\n");
    }
    return own;
  }

  for (int i = 0; i < 5; ++i) {
    const std::string d = g4particlexs_subdir(rows[i].dir);
    auto* t = new data::ParticleXsTable<real_t>();
    hxs::PxsDataSet<real_t> ds;
    if (!hxs::pxs_load<real_t>(hxs::PxsKind::kParticleInelastic, rows[i].proj(), d, *t, ds)) {
      std::printf("\nFATAL: G4PARTICLEXS4.0/%s is incomplete - an element file is missing.\n"
                  "  A missing dataset is fatal here as everywhere (src/host/g4data.cuh),\n"
                  "  because a hadron with a partial cross section is a transport that is\n"
                  "  quietly wrong in one element.\n", rows[i].dir);
      std::exit(1);
    }
    own.view.particle[i] = detail::upload_pxs<real_t>(*t, ds, own);
    ++own.n_pxs;
    delete t;
  }

  // G4BGGPionInelasticXS, both pion charges in one table. `is_elastic = false` selects the
  // inelastic arrays throughout - `theLowEPiPlus`/`theLowEPiMinus` rather than the elastic
  // pair, and the 20 MeV boundary on the other side (see `xs/bgg_pion_xs.cuh`'s header, which
  // records that at exactly 20 MeV the two classes take different branches).
  {
    auto* h = new hxs::BggPionTable<real_t>();
    hxs::bgg_build_pion_table<real_t>(/*is_elastic=*/false, *h);
    void* p = nullptr;
    if (cudaMalloc(&p, sizeof(*h)) != cudaSuccess
        || cudaMemcpy(p, h, sizeof(*h), cudaMemcpyHostToDevice) != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of G4BGGPionInelasticXS\n", sizeof(*h));
      std::exit(1);
    }
    own.allocs.push_back(p);
    own.bytes += sizeof(*h);
    own.view.bgg_pion = static_cast<const hxs::BggPionTable<real_t>*>(p);
    delete h;
  }

  own.ok = true;
  if (verbose) {
    std::printf("inelastic tables: %.2f MB - %d G4ParticleInelasticXS data sets "
                "(p, d, t, He3, alpha), G4BGGPionInelasticXS %zu B\n",
                double(own.bytes) / 1048576.0, own.n_pxs, sizeof(hxs::BggPionTable<real_t>));
  }
  return own;
}

template <typename real_t>
inline void free_inelastic_tables(InelasticTableOwner<real_t>& own) {
  for (void* p : own.allocs) { cudaFree(p); }
  own = InelasticTableOwner<real_t>{};
}

// =============================================================================================
// P19: G4EmExtraPhysics' tables - the photon's, the leptons' and the muon's nuclear processes
// =============================================================================================
//
// Three tables, each built once exactly where Geant4 builds it once, and the per-material
// thresholds `had::emextra_xs_per_volume` skips below:
//
//   G4GammaNuclearXS          G4PARTICLEXS4.0/gamma, 94 element and ~190 isotope vectors -
//                             `G4GammaNuclearXS::Initialise`. P2's reader, P13's CHIPS above it.
//   G4KokoulinMuonNuclearXS   the 61-node log vector per Z from 1 GeV to 1 PeV -
//                             `BuildCrossSectionTable`, P13's eight-point Gauss-Legendre.
//   G4MuonVDNuclearModel      its 5 x 73 x 800 cumulative sampling table - `MakeSamplingTable`,
//                             in the model's constructor. 2.3 MB, read-only, one copy.
//
// The two CHIPS parameterisations (photo- and electro-nuclear) are function-scope constant
// tables compiled into the device code and need nothing here.
//
// BUILT UNCONDITIONALLY, like the inelastic tables above, because `Upload` cannot know whether
// a photon beam will ever make a photon above 2 m_e - and a gamma run with no photo-nuclear table
// would be a photon with no nuclear process rather than a failure anyone would see. The cost is
// printed. A run whose G4PARTICLEXSDATA cannot be resolved gets no photo-nuclear table and says
// so; the two lepton processes need no data set at all.

/// The last leading-zero node of one element's `G4GammaNuclearXS` vector: at and below it the
/// element's cross section is EXACTLY zero, because `G4PhysicsVector::Value` interpolates a
/// straight line (no spline - `xs/physics_vector.cuh`'s header) between two zeros and clamps to
/// the first value below the first node. Zero when the first value is not zero, or when the
/// element has no vector (then the cross section is CHIPS's or a refusal, and every energy is
/// evaluated).
template <typename real_t>
inline real_t emextra_gamma_element_threshold(const hadronic::xs::PxsDataSet<real_t>& ds, int z) {
  namespace hxs = g4gpu::hadronic::xs;
  const int zz = (z >= hxs::pxs_maxz(ds.kind)) ? hxs::pxs_maxz(ds.kind) - 1 : z;
  const hxs::PhysVec<real_t> pv = hxs::pxs_view<real_t>(*ds.data, ds.data->element[zz]);
  if (pv.empty()) { return real_t(0); }
  int last0 = -1;
  for (int i = 0; i < pv.n; ++i) {
    if (pv.value_at(i) == real_t(0)) {
      last0 = i;
    } else {
      break;
    }
  }
  return (last0 >= 0) ? pv.energy(last0) : real_t(0);
}

/// `G4ElectroNuclearCrossSection::GetElementCrossSection`'s two zero returns, as one energy:
/// `Energy <= EMi` (2.0612 MeV) and `Energy <= ThresholdEnergy(Z, N)`, with N = int(A) - Z off
/// the NIST mean atomic mass exactly as the class computes it. An element the class refuses
/// (Z outside 1..98) gets zero, i.e. "evaluate", so its refusal is reached rather than skipped.
template <typename real_t>
inline real_t emextra_electro_element_threshold(int z) {
  namespace chips = g4gpu::hadronic::xs::chips;
  bool refused = false;
  const chips::ElnNucleus n = chips::eln_nucleus(z, refused);
  if (refused) { return real_t(0); }
  const double th = (n.TH > chips::eln_EMi()) ? n.TH : chips::eln_EMi();
  return static_cast<real_t>(th);
}

/// P13's tables on the HOST, owned: what `upload_emextra_tables` copies to the device, and what
/// the host tests hand the transport's own functions directly.
template <typename real_t>
struct EmExtraHostTables {
  data::ParticleXsTable<real_t>* gamma_table = nullptr;
  hadronic::xs::PxsDataSet<real_t> gamma{};
  hadronic::xs::kokoulin::KokoulinTable<real_t>* kokoulin = nullptr;
  physics::hadronic::emextra::MuVdTable* mu_vd = nullptr;
  /// `had::kNumEmExtraThresholds * n_mat` energies, row-major by kind - `had::EmExtraTables`'s.
  std::vector<real_t> threshold;
  /// `had::kNumEmExtraThresholds * had::kEmExtraMaxZ` energies, the same per element.
  std::vector<real_t> element_threshold;
  int n_mat = 0;
  bool gamma_ok = false;
  /// P21: `G4GammaGeneralProcess`'s zone-2 table 9 - the 51 node energies, `n_mat` x 51 values
  /// and the per-material energy at and below which it is exactly 1 - and the view of them.
  /// Empty without the gamma data set or without the photoelectric table it is built from.
  std::vector<real_t> p9_e, p9_v, p9_threshold;
  had::GammaGeneralTable9<real_t> p9{};
  bool p9_ok = false;
  /// P26: every table of the general process, built as `BuildPhysicsTable` builds them - the four
  /// grids, the values and second derivatives of each slot, `n_mat` rows each - and the
  /// sub-processes' own lambda tables they were built from, `n_mat` x 4 (`em::GammaSubProcess`
  /// order), kept for the host tests. Empty without the photoelectric or the Rayleigh table.
  bool gg_ok = false;
  std::vector<real_t> gg_e[4];
  std::vector<real_t> gg_t[had::kGammaGeneralNumTables];
  std::vector<real_t> gg_d2[had::kGammaGeneralNumTables];
  std::vector<em::EmProcessLambda> gg_sub;

  /// A view whose pointers are HOST pointers, for host callers of the transport's functions.
  had::EmExtraTables<real_t> view() const {
    had::EmExtraTables<real_t> v;
    v.gamma = gamma_ok ? &gamma : nullptr;
    v.kokoulin = kokoulin;
    v.mu_vd = mu_vd;
    v.threshold = threshold.empty() ? nullptr : threshold.data();
    v.n_materials = n_mat;
    v.element_threshold = element_threshold.empty() ? nullptr : element_threshold.data();
    v.p9 = p9_ok ? &p9 : nullptr;
    return v;
  }
  EmExtraHostTables() = default;
  EmExtraHostTables(const EmExtraHostTables&) = delete;
  EmExtraHostTables& operator=(const EmExtraHostTables&) = delete;
  ~EmExtraHostTables() {
    delete gamma_table;
    delete kokoulin;
    delete mu_vd;
  }
};

/// `G4GammaGeneralProcess::InitialiseProcess`'s four grids (G4GammaGeneralProcess.cc:213-223):
///
///     nbin1 = max(5, nd*G4lrint(log10(minPEEnergy/mine)))      21
///     nbin2 = max(5, nd*G4lrint(log10(maxe/minMMEnergy)))      42
///     aVector(mine, minPEEnergy, nbin1, true)                  zone 0, splined
///     bVector(minPEEnergy, minEEEnergy, nLowE = 40, false)     zone 1
///     cVector(minEEEnergy, minMMEnergy, nHighE = 50, false)    zone 2 (P21's grid)
///     dVector(minMMEnergy, maxe, nbin2, true)                  zone 3, splined
///
/// with `mine`, `maxe` and `nd` the EM parameters' defaults. The edges are the same doubles
/// `had::gamma_general_min_*` return; `minPEEnergy` is `150*CLHEP::keV`.
inline void gamma_general_grids(em::LogVecHost g[4]) {
  const double mine = em::kEmParamMinKinEnergy, maxe = em::kEmParamMaxKinEnergy;
  const double min_pe = 150 * em::kEmKeV;
  const double min_ee = had::gamma_general_min_ee<double>();
  const double min_mm = had::gamma_general_min_mm<double>();
  const int nd = em::kEmParamBinsPerDecade;
  int nbin1 = nd * em::g4_lrint(std::log10(min_pe / mine));
  if (nbin1 < 5) { nbin1 = 5; }
  int nbin2 = nd * em::g4_lrint(std::log10(maxe / min_mm));
  if (nbin2 < 5) { nbin2 = 5; }
  g[0].make(mine, min_pe, nbin1, true);
  g[1].make(min_pe, min_ee, 40, false);
  g[2].make(min_ee, min_mm, had::kGammaGeneralZone2Bins, false);
  g[3].make(min_mm, maxe, nbin2, true);
}

/// P26: `G4GammaGeneralProcess::BuildPhysicsTable` for @p n_mat materials, into @p h.
///
/// First each sub-process's own `BuildPhysicsTable` - `em::EmProcessLambda::build`, its lambda
/// and prim vectors from this port's models - then the four zones' loops in the source's order of
/// operations (G4GammaGeneralProcess.cc:309-438), each node's sigmas through `GetLambda`
/// (`em::EmProcessLambda::get_lambda`) and sigN through photonNuclear's data store
/// (`gn->ComputeCrossSection`, which is `had::emextra_xs_per_volume`), and last
/// `FillSecondDerivatives` on tables 0, 1 and 10-14.
///
/// The models, as each process's `InitialiseProcess` sets them up:
///   * Compton: `G4KleinNishinaCompton` with `SetLowEnergyLimit(param->MinKinEnergy())` - so its
///     `ComputeCrossSectionPerAtom` returns 0 at `GammaEnergy <= 100 eV`, where this port's
///     `em::compton_xs_per_atom` keeps G4VEmModel's 10 eV default (the photon is absorbed at
///     990 eV, so the transport never asks). Only node 0 of Compton's lambda vector sits there,
///     and Geant4 stores 0 at it (`ref/oracle/gammagp_tables.csv`, table 300).
///   * conversion: `G4PairProductionRelModel` - this port's `em::pair_xs_per_atom`, the
///     parameterisation Geant4 uses below 30 GeV. Above 30 GeV Geant4 integrates the
///     differential cross section with LPM and this port does not (docs/PORTED.md 4.1), so
///     zone 3's nodes above 30 GeV carry that difference, and through the spline's tridiagonal
///     solve a vanishing amount of it below; the host test measures how much.
///   * photoelectric: `G4LivermorePhotoElectricModel`, `data::photoelectric_xs_per_atom`.
///   * Rayleigh: `G4LivermoreRayleighModel`, `data::rayleigh_xs_per_atom`.
/// Each summed over the material's elements in their order, `G4VEmModel::CrossSectionPerVolume`.
template <typename real_t>
inline void build_gamma_general_host_tables(EmExtraHostTables<real_t>& h,
                                            const data::Material<real_t>* mats, int n_mat,
                                            const data::PhotoElectricTable<real_t>& pe,
                                            const data::RayleighTable<real_t>& ray,
                                            bool read_tables) {
  namespace E = g4gpu::em;
  const double me = units::electron_mass_c2<double>();
  em::LogVecHost grid[4];
  gamma_general_grids(grid);
  for (int z = 0; z < 4; ++z) { h.gg_e[z].assign(grid[z].e.begin(), grid[z].e.end()); }
  for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
    h.gg_t[k].clear();
    h.gg_d2[k].clear();
  }
  h.gg_sub.assign(static_cast<std::size_t>(n_mat) * 4, E::EmProcessLambda{});
  const had::EmExtraTables<real_t> hv = h.view();

  for (int m = 0; m < n_mat; ++m) {
    const data::Material<real_t>& mat = mats[m];
    auto compt = [&](double e) -> double {
      if (e <= E::kEmParamMinKinEnergy) { return 0.0; }
      return static_cast<double>(
          em::gamma_macroscopic_xs<real_t>(mat, real_t(e), nullptr, nullptr, true, false).compton);
    };
    // `SetupForMaterial`: `fLPMEnergy = mat->GetRadlen()*gLPMconstant`, then the element sum,
    // through the model's whole range (`em::pair_xs_per_atom_rel`).
    const real_t lpm_energy = mat.radiation_length * em::lpm_constant<real_t>();
    auto conv = [&](double e) -> double {
      real_t s = real_t(0);
      for (int i = 0; i < mat.n_elements; ++i) {
        s += mat.n_atoms[i] * em::pair_xs_per_atom_rel<real_t>(real_t(e), mat.z[i], lpm_energy);
      }
      return static_cast<double>(s);
    };
    auto phot = [&](double e) -> double {
      return static_cast<double>(
          em::gamma_macroscopic_xs<real_t>(mat, real_t(e), &pe, nullptr, false, false)
              .photoelectric);
    };
    auto rayl = [&](double e) -> double {
      return static_cast<double>(
          em::gamma_macroscopic_xs<real_t>(mat, real_t(e), nullptr, &ray, false, false)
              .rayleigh);
    };
    E::EmProcessLambda* sub = &h.gg_sub[static_cast<std::size_t>(m) * 4];
    // `BuildPhysicsTable`'s order: photoelectric, Compton, conversion, Rayleigh. The order does
    // not change a number - each process's tables are its own - but it is the source's.
    sub[2].build(E::em_process_lambda_config(E::GammaSubProcess::kPhotoelectric, me), phot);
    sub[0].build(E::em_process_lambda_config(E::GammaSubProcess::kCompton, me), compt);
    sub[1].build(E::em_process_lambda_config(E::GammaSubProcess::kConversion, me), conv);
    sub[3].build(E::em_process_lambda_config(E::GammaSubProcess::kRayleigh, me), rayl);
    auto L = [&](int k, double e, double loge) -> double {
      switch (k) {
        case 0: return sub[0].get_lambda(e, loge, compt);
        case 1: return sub[1].get_lambda(e, loge, conv);
        case 2: return sub[2].get_lambda(e, loge, phot);
        default: return sub[3].get_lambda(e, loge, rayl);
      }
    };
    auto sig_n_at = [&](double e) -> double {
      hadronic::xs::MaterialXs<real_t> mxs{};
      return static_cast<double>(had::emextra_xs_per_volume<real_t>(
          hv, had::EmExtraProcess::kPhotonNuclear, m, mat, real_t(e), mxs));
    };
    // Per-material vectors, one per slot, filled node by node as the source fills them.
    std::vector<double> tv[had::kGammaGeneralNumTables];
    for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
      if (k == 5) { continue; }
      tv[k].assign(grid[had::gamma_general_table_zone(k)].e.size(), 0.0);
    }
    // energy interval 0
    for (std::size_t j = 0; j < grid[0].e.size(); ++j) {
      const double e = grid[0].e[j];
      const double loge = std::log(e);
      const double sigComp = L(0, e, loge);
      const double sigR = L(3, e, loge);
      const double sum = sigComp + sigR;
      tv[0][j] = sum;
      tv[1][j] = sigR / sum;
    }
    // energy interval 1
    for (std::size_t j = 0; j < grid[1].e.size(); ++j) {
      const double e = grid[1].e[j];
      const double loge = std::log(e);
      const double sigComp = L(0, e, loge);
      const double sigR = L(3, e, loge);
      const double sigPE = L(2, e, loge);
      const double sum = sigComp + sigR + sigPE;
      tv[2][j] = sum;
      tv[3][j] = sigPE / sum;
      tv[4][j] = (sigR > 0.0) ? (sigComp + sigPE) / sum : 1.0;
    }
    // energy interval 2
    for (std::size_t j = 0; j < grid[2].e.size(); ++j) {
      const double e = grid[2].e[j];
      const double loge = std::log(e);
      const double sigComp = L(0, e, loge);
      const double sigConv = L(1, e, loge);
      const double sigPE = L(2, e, loge);
      const double sigN = sig_n_at(e);
      const double sum = sigComp + sigConv + sigPE + sigN;
      tv[6][j] = sum;
      tv[7][j] = sigConv / sum;
      tv[8][j] = (sigConv + sigComp) / sum;
      tv[9][j] = (sigN > 0.0) ? (sigConv + sigComp + sigPE) / sum : 1.0;
    }
    // energy interval 3. `sigM` is G4GammaConversionToMuons', which QBBC does not build: 0.0, and
    // written into every sum and share where the source writes it, since `x + 0.0` is `x`.
    for (std::size_t j = 0; j < grid[3].e.size(); ++j) {
      const double e = grid[3].e[j];
      const double loge = std::log(e);
      const double sigComp = L(0, e, loge);
      const double sigConv = L(1, e, loge);
      const double sigPE = L(2, e, loge);
      const double sigN = sig_n_at(e);
      const double sigM = 0.0;
      const double sum = sigComp + sigConv + sigPE + sigN + sigM;
      tv[10][j] = sum;
      tv[11][j] = (sigComp + sigPE + sigN + sigM) / sum;
      tv[12][j] = (sigPE + sigN + sigM) / sum;
      tv[13][j] = (sigN + sigM) / sum;
      tv[14][j] = sigN / sum;
    }
    for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
      if (k == 5) { continue; }
      const em::LogVecHost& g = grid[had::gamma_general_table_zone(k)];
      for (const double x : tv[k]) { h.gg_t[k].push_back(static_cast<real_t>(x)); }
      if (had::gamma_general_table_spline(k)) {
        std::vector<double> d(tv[k].size(), 0.0);
        data::fill_second_derivatives(g.e.data(), tv[k].data(), static_cast<int>(tv[k].size()),
                                      d.data());
        for (const double x : d) { h.gg_d2[k].push_back(static_cast<real_t>(x)); }
      }
    }
  }

  // The device-shaped view, HOST pointers; `upload_emextra_tables` swaps them for device copies.
  for (int z = 0; z < 4; ++z) {
    had::GammaGeneralGrid<real_t>& g = h.p9.grid[z];
    g.e = h.gg_e[z].data();
    g.n = static_cast<int>(h.gg_e[z].size());
    g.edge_min = static_cast<real_t>(grid[z].edge_min);
    g.edge_max = static_cast<real_t>(grid[z].edge_max);
    g.inv_dbin = static_cast<real_t>(grid[z].inv_dbin);
    g.log_emin = static_cast<real_t>(grid[z].log_emin);
  }
  for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
    h.p9.t[k] = h.gg_t[k].empty() ? nullptr : h.gg_t[k].data();
    h.p9.d2[k] = h.gg_d2[k].empty() ? nullptr : h.gg_d2[k].data();
  }
  h.p9.built_with_nuclear = h.gamma_ok ? 1 : 0;
  h.p9.read_tables = read_tables ? 1 : 0;
  h.gg_ok = true;
}

/// `G4GPU_GAMMA_GENERAL_TABLES`: unset or anything but "0" reads the general process's tables
/// (P26, as Geant4 does); "0" keeps P1's models and P21's table 9, the comparison's other side.
inline bool gamma_general_tables_from_env() {
  const char* s = std::getenv("G4GPU_GAMMA_GENERAL_TABLES");
  return !(s != nullptr && s[0] == '0' && s[1] == '\0');
}

/// Builds P13's three tables on the host and the thresholds of @p n_mat materials - and since
/// P21 the general process's table 9 from @p pe.
///
/// The muon's two tables are built with the mu- mass for both charges: `G4KokoulinMuonNuclearXS`
/// reads `G4MuonMinus::MuonMinus()->GetPDGMass()` unconditionally (P13's note in
/// `xs/kokoulin_muon_xs.cuh`), and one `G4MuonVDNuclearModel` serves both muons.
///
/// @param pe the photoelectric table with HOST pointers, for every element of @p mats. Table 9's
///        nodes are `(sigConv + sigComp + sigPE)/sum`, so without it there is no table 9 - and
///        then no photo-nuclear term in zone 2 at all, which this prints when the gamma data
///        set is there to be missed.
template <typename real_t>
inline void build_emextra_host_tables(EmExtraHostTables<real_t>& h,
                                      const data::Material<real_t>* mats, int n_mat,
                                      bool verbose = true,
                                      const data::PhotoElectricTable<real_t>* pe = nullptr,
                                      const data::RayleighTable<real_t>* ray = nullptr,
                                      bool read_tables = gamma_general_tables_from_env()) {
  namespace hxs = g4gpu::hadronic::xs;
  namespace ee = g4gpu::physics::hadronic::emextra;
  const std::string gdir = g4particlexs_subdir("gamma");
  if (!gdir.empty()) {
    h.gamma_table = new data::ParticleXsTable<real_t>();
    if (!hxs::pxs_load<real_t>(hxs::PxsKind::kGammaNuclear, hxs::gamma<real_t>(), gdir,
                               *h.gamma_table, h.gamma)) {
      std::printf("\nFATAL: G4PARTICLEXS4.0/gamma is incomplete - an element file is missing.\n"
                  "  A missing dataset is fatal here as everywhere (src/host/g4data.cuh): a\n"
                  "  photon with a partial photo-nuclear cross section is quietly wrong in one\n"
                  "  element.\n");
      std::exit(1);
    }
    h.gamma_ok = true;
  } else if (verbose) {
    std::printf("photo-nuclear table: G4PARTICLEXSDATA could not be resolved - the photon has "
                "no nuclear process\n");
  }
  const real_t mu_mass = particle_def<real_t>(ParticleType::kMuonMinus).mass;
  h.kokoulin = new hxs::kokoulin::KokoulinTable<real_t>();
  hxs::kokoulin::build_table<real_t>(*h.kokoulin, mu_mass);
  h.mu_vd = new ee::MuVdTable();
  ee::mu_vd_make_sampling_table(*h.mu_vd, static_cast<double>(mu_mass), ee::g_per_mole());

  // The thresholds: the smallest over a material's elements, because the material's cross
  // section is zero only where EVERY element's is.
  h.n_mat = n_mat;
  h.threshold.assign(static_cast<std::size_t>(had::kNumEmExtraThresholds) * n_mat, real_t(0));
  for (int m = 0; m < n_mat; ++m) {
    real_t tg = real_t(1e30), te = real_t(1e30);
    for (int i = 0; i < mats[m].n_elements; ++i) {
      const int z = static_cast<int>(mats[m].z[i] + real_t(0.5));
      const real_t g = h.gamma_ok ? emextra_gamma_element_threshold<real_t>(h.gamma, z)
                                  : real_t(0);
      const real_t e = emextra_electro_element_threshold<real_t>(z);
      if (g < tg) { tg = g; }
      if (e < te) { te = e; }
    }
    if (mats[m].n_elements <= 0) { tg = te = real_t(0); }
    h.threshold[static_cast<std::size_t>(had::kGammaNuclearThreshold) * n_mat + m] = tg;
    h.threshold[static_cast<std::size_t>(had::kElectroNuclearThreshold) * n_mat + m] = te;
  }
  // And per element, for every Z either class answers - the same two functions, so the material
  // row above is exactly the smallest of its elements' entries here.
  h.element_threshold.assign(
      static_cast<std::size_t>(had::kNumEmExtraThresholds) * had::kEmExtraMaxZ, real_t(0));
  for (int z = 1; z < had::kEmExtraMaxZ; ++z) {
    h.element_threshold[static_cast<std::size_t>(had::kGammaNuclearThreshold) * had::kEmExtraMaxZ
                        + z] = h.gamma_ok ? emextra_gamma_element_threshold<real_t>(h.gamma, z)
                                          : real_t(0);
    h.element_threshold[static_cast<std::size_t>(had::kElectroNuclearThreshold) * had::kEmExtraMaxZ
                        + z] = emextra_electro_element_threshold<real_t>(z);
  }

  // P21: G4GammaGeneralProcess's table 9, at Geant4's own 51 nodes, from this port's cross
  // sections - `had::gamma_general_table9_node` is the loop body of `BuildPhysicsTable`'s zone 2.
  // Built from the HOST view above, so the store it evaluates is the one the drain will use.
  // P26: built without the gamma data set too - every node is then 1.0, its threshold the last
  // node, and nothing reads it - because since P26 the struct carries every other table as well.
  h.p9_ok = false;
  if (pe != nullptr && n_mat > 0) {
    h.p9_e.assign(had::kGammaGeneralZone2Nodes, real_t(0));
    real_t inv_dbin = 0, log_emin = 0;
    had::gamma_general_zone2_grid<real_t>(h.p9_e.data(), inv_dbin, log_emin);
    h.p9_v.assign(static_cast<std::size_t>(n_mat) * had::kGammaGeneralZone2Nodes, real_t(1));
    h.p9_threshold.assign(static_cast<std::size_t>(n_mat), real_t(0));
    const had::EmExtraTables<real_t> hv = h.view();
    for (int m = 0; m < n_mat; ++m) {
      int last_one = -1;
      bool leading = true;
      for (int j = 0; j < had::kGammaGeneralZone2Nodes; ++j) {
        const real_t v = had::gamma_general_table9_node<real_t>(hv, m, mats[m], *pe, h.p9_e[j]);
        h.p9_v[static_cast<std::size_t>(m) * had::kGammaGeneralZone2Nodes + j] = v;
        if (leading && v == real_t(1)) {
          last_one = j;
        } else {
          leading = false;
        }
      }
      // At and below the last leading 1.0 node the interpolant is 1.0 to the bit (`y1 + b*dy`
      // with y1 = 1, dy = 0), and at or below `edgeMin` it is node 0's value: either way exactly
      // 1, so the stepper may skip the evaluation there. Above it, it may not.
      h.p9_threshold[m] = (last_one >= 0) ? h.p9_e[last_one] : real_t(0);
    }
    h.p9.e = h.p9_e.data();
    h.p9.v = h.p9_v.data();
    h.p9.threshold = h.p9_threshold.data();
    h.p9.n_materials = n_mat;
    h.p9.edge_min = h.p9_e[0];
    h.p9.edge_max = h.p9_e[had::kGammaGeneralZone2Bins];
    h.p9.inv_dbin = inv_dbin;
    h.p9.log_emin = log_emin;
    h.p9_ok = true;
  } else if (h.gamma_ok && verbose) {
    std::printf("photo-nuclear table 9: no photoelectric table was given to build it from - "
                "the photon has no photo-nuclear term in zone 2 (2 m_e to 100 MeV)\n");
  }
  // P26: the rest of the general process's tables, which need the Rayleigh table as well.
  h.gg_ok = false;
  if (h.p9_ok && ray != nullptr) {
    build_gamma_general_host_tables<real_t>(h, mats, n_mat, *pe, *ray, read_tables);
  } else if (verbose) {
    std::printf("G4GammaGeneralProcess tables: no photoelectric or Rayleigh table to build them "
                "from - the photon reads its models (P1) and table 9 alone\n");
  }
}

/// Everything `upload_emextra_tables` allocated.
template <typename real_t>
struct EmExtraTableOwner {
  had::EmExtraTables<real_t> view{};
  std::vector<void*> allocs;
  std::size_t bytes = 0;
  bool gamma_ok = false;
};

/// Builds P13's tables on the host and uploads them, with the thresholds of the scene's
/// materials and, since P21, the general process's table 9. The view it returns travels in
/// `had::HadronicWiring::emextra`.
///
/// @param pe the photoelectric table with HOST pointers (`data::load_photoelectric`'s), which
///        table 9 is built from. Required for a photo-nuclear term in zone 2, and so not
///        defaulted: a caller that forgot it would get a photon that silently never reacts
///        below 100 MeV.
template <typename real_t>
inline EmExtraTableOwner<real_t> upload_emextra_tables(
    const data::Material<real_t>* mats, int n_mat, bool verbose,
    const data::PhotoElectricTable<real_t>* pe) {
  namespace hxs = g4gpu::hadronic::xs;
  namespace ee = g4gpu::physics::hadronic::emextra;
  EmExtraTableOwner<real_t> own;
  EmExtraHostTables<real_t> h;
  // P26: the Rayleigh cross sections on the HOST, for the scene's elements - zones 0 and 1 of the
  // general process sum Rayleigh's `GetLambda`. Read here, from the directory
  // `host::default_rayl_dir` names (G4GPU_RAYL_DIR, else G4EMLOW's epics2017/rayl), rather than
  // passed in, so that the engine's upload needed no new argument. A missing file is fatal, as
  // the device upload of the same data is.
  data::RayleighTable<real_t> ray_host{};
  std::vector<real_t> ray_e, ray_v;
  const data::RayleighTable<real_t>* ray = nullptr;
  if (pe != nullptr) {
    std::vector<int> zs;
    for (int m = 0; m < n_mat; ++m) {
      for (int i = 0; i < mats[m].n_elements; ++i) {
        const int z = static_cast<int>(mats[m].z[i] + real_t(0.5));
        bool seen = false;
        for (const int y : zs) { seen |= (y == z); }
        if (!seen) { zs.push_back(z); }
      }
    }
    const char* env = std::getenv("G4GPU_RAYL_DIR");
    const std::string rdir = env ? std::string(env) : g4emlow_subdir("epics2017/rayl", "re-cs-1.dat");
    if (!data::load_rayleigh<real_t>(rdir, zs.data(), static_cast<int>(zs.size()), ray_host,
                                     ray_e, ray_v)) {
      std::printf("\nFATAL: the Rayleigh data (G4EMLOW epics2017/rayl) could not be read from "
                  "'%s' for the general process's tables\n", rdir.c_str());
      std::exit(1);
    }
    ray_host.table_e = ray_e.data();
    ray_host.table_v = ray_v.data();
    ray_host.table_n = static_cast<int>(ray_e.size());
    ray = &ray_host;
  }
  build_emextra_host_tables<real_t>(h, mats, n_mat, verbose, pe, ray);

  auto up = [&own](const void* src, std::size_t n) -> void* {
    void* p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess || cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)
                                                 != cudaSuccess) {
      std::printf("\nFATAL: could not upload %zu bytes of a G4EmExtraPhysics table\n", n);
      std::exit(1);
    }
    own.allocs.push_back(p);
    own.bytes += n;
    return p;
  };

  if (h.gamma_ok) {
    own.view.gamma = detail::upload_pxs<real_t>(*h.gamma_table, h.gamma, own);
    own.gamma_ok = true;
  }
  own.view.kokoulin = static_cast<const hxs::kokoulin::KokoulinTable<real_t>*>(
      up(h.kokoulin, sizeof(*h.kokoulin)));
  own.view.mu_vd = static_cast<const ee::MuVdTable*>(up(h.mu_vd, sizeof(ee::MuVdTable)));
  if (!h.threshold.empty()) {
    own.view.threshold =
        static_cast<const real_t*>(up(h.threshold.data(), h.threshold.size() * sizeof(real_t)));
  }
  own.view.n_materials = n_mat;
  if (!h.element_threshold.empty()) {
    own.view.element_threshold = static_cast<const real_t*>(
        up(h.element_threshold.data(), h.element_threshold.size() * sizeof(real_t)));
  }
  // P21: table 9 - its three arrays, then the struct pointing at the device copies.
  if (h.p9_ok) {
    had::GammaGeneralTable9<real_t> d9 = h.p9;
    d9.e = static_cast<const real_t*>(up(h.p9_e.data(), h.p9_e.size() * sizeof(real_t)));
    d9.v = static_cast<const real_t*>(up(h.p9_v.data(), h.p9_v.size() * sizeof(real_t)));
    d9.threshold = static_cast<const real_t*>(
        up(h.p9_threshold.data(), h.p9_threshold.size() * sizeof(real_t)));
    // P26: the four grids and every built slot's values and second derivatives.
    if (h.gg_ok) {
      for (int z = 0; z < 4; ++z) {
        d9.grid[z].e =
            static_cast<const real_t*>(up(h.gg_e[z].data(), h.gg_e[z].size() * sizeof(real_t)));
      }
      for (int k = 0; k < had::kGammaGeneralNumTables; ++k) {
        d9.t[k] = h.gg_t[k].empty() ? nullptr
                                    : static_cast<const real_t*>(up(
                                          h.gg_t[k].data(), h.gg_t[k].size() * sizeof(real_t)));
        d9.d2[k] = h.gg_d2[k].empty() ? nullptr
                                      : static_cast<const real_t*>(up(
                                            h.gg_d2[k].data(), h.gg_d2[k].size() * sizeof(real_t)));
      }
    }
    own.view.p9 = static_cast<const had::GammaGeneralTable9<real_t>*>(up(&d9, sizeof(d9)));
  }

  if (verbose) {
    std::printf("photo-/lepto-nuclear tables: %.2f MB - G4GammaNuclearXS %s, "
                "G4KokoulinMuonNuclearXS %zu B, G4MuonVDNuclearModel sampling table %zu B, "
                "G4GammaGeneralProcess table 9 %s, its other tables %s\n",
                double(own.bytes) / 1048576.0, h.gamma_ok ? "loaded" : "ABSENT",
                sizeof(hxs::kokoulin::KokoulinTable<real_t>), sizeof(ee::MuVdTable),
                h.p9_ok ? "built" : "ABSENT",
                !h.gg_ok ? "ABSENT" : (h.p9.read_tables ? "built and READ" : "built, NOT read "
                                                          "(G4GPU_GAMMA_GENERAL_TABLES=0)"));
  }
  return own;
}

template <typename real_t>
inline void free_emextra_tables(EmExtraTableOwner<real_t>& own) {
  for (void* p : own.allocs) { cudaFree(p); }
  own = EmExtraTableOwner<real_t>{};
}

}  // namespace g4gpu::host
