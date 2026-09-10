// The five G4PARTICLEXS data sets, their isotope branch, and THE NEUTRON GRID.
//
// Three oracle files:
//
//   had_particlexs.csv       every element cross section of the five data sets - proton,
//                            deuteron, triton, He3 and alpha inelastic; neutron elastic,
//                            inelastic and capture; gamma-nuclear - twelve points per decade
//                            over each set's own range, Z = 1..92 (94 for gamma), with the
//                            hand-over energies added exactly.
//   had_particlexs_iso.csv   the same sets per isotope, for every (Z, A) with a data file, at
//                            sixteen energies chosen to straddle the isotope window's edge
//                            rather than to be many.
//   had_neutron_general.csv  G4NeutronGeneralProcess's five tables, read back out of the
//                            process's own StorePhysicsTable output - Geant4's binVector, not
//                            a formula. With had_matelem.csv, which carries each material's
//                            elements and atom densities, because those tables are
//                            macroscopic.
//
// WHY THE GENERAL PROCESS'S TABLE IS THE POINT OF THIS FILE
//
// docs/RISK.md V5: Geant4 transports a TABLE built from the model, not the model.
// ref/oracle/hadronic_params.csv says EnableNeutronGeneralProcess = 1, so a neutron in QBBC
// has one discrete process whose interaction length comes from a 401-point log vector per
// material over 1 keV to 20 MeV and a 71-point one over 20 MeV to 100 TeV. Getting
// G4NeutronElasticXS exactly right and then evaluating it per step would be right about the
// cross section and wrong about the transport. So the three data sets are checked first,
// element and isotope, and then the table they are folded into is checked against the table
// Geant4 actually built - grid nodes included, since a wrong node count puts every interior
// point in a different place while leaving the two ends exact.
//
// AND WHY TWO BAD DATASET FILES ARE TESTED ON PURPOSE
//
// Geant4 treats the two ways of failing to read one file DIFFERENTLY, and the difference is
// easy to miss because it is a `warn` flag on one branch and no flag on the other: an element
// file that will not OPEN is a FatalException, and ANY file that opens and will not PARSE is a
// FatalException whether it is an element or an isotope. A MISSING isotope file is legitimate -
// not every A in [amin, amax] ships one - and the readers fall back from it.
//
// The failure mode both guards is not a crash but its opposite: a reader that returns zero for
// a missing element gives a particle that never interacts there, and one that treats a corrupt
// isotope file as absent gives the element cross section times A/aeff[Z] - a plausible number
// from a file Geant4 would have aborted on. The last section stages both directories and
// requires the loader to refuse both.
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "data/isotope_list.hh"
#include "data/materials.cuh"
#include "data/particlexs_data.cuh"
#include "host/g4data.cuh"
#include "physics/hadronic/xs/neutron_general_xs.cuh"
#include "physics/hadronic/xs/particlexs.cuh"
#include "physics/hadronic/xs/refusal.cuh"
#include "physics/hadronic/xs/sample_za.cuh"

using namespace g4gpu;
using namespace g4gpu::hadronic::xs;
using real_t = double;

namespace {

constexpr double kTol = 1e-12;

struct Cell {
  int n = 0;
  double worst = 0;
  std::string where;
  int refused = 0;
};

void note(Cell& c, double dev, const char* what) {
  ++c.n;
  if (dev > c.worst) {
    c.worst = dev;
    c.where = what;
  }
}

double deviation(double ours, double g4) {
  const double floor = 1e-300;
  if (std::fabs(g4) > floor) { return std::fabs(ours - g4) / std::fabs(g4); }
  return (std::fabs(ours) > floor) ? 1.0 : 0.0;
}

void cmp(Cell& c, double ours, double g4, const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  char full[420];
  std::snprintf(full, sizeof full, "%s (ours %.17g, G4 %.17g)", buf, ours, g4);
  note(c, deviation(ours, g4), full);
}

FILE* open_oracle(const std::string& dir, const char* name) {
  const std::string path = dir + "/" + name;
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) {
    std::printf("FAIL: cannot read %s - run ref/oracle/run.bat tables in this worktree "
                "first\n", path.c_str());
    return nullptr;
  }
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return nullptr;
  }
  return f;
}

/// The nine (dataset, particle) pairs the dump emits, in its order. `dir` is the
/// G4PARTICLEXS subdirectory, which is the Geant4 particle name and not a mapping.
struct SetSpec {
  const char* dataset;   ///< the dump's `dataset` column
  const char* particle;  ///< the dump's `particle` column
  const char* dir;       ///< the G4PARTICLEXS subdirectory
  PxsKind kind;
};

const SetSpec kSets[] = {
    {"ParticleInelastic", "proton", "proton", PxsKind::kParticleInelastic},
    {"ParticleInelastic", "deuteron", "deuteron", PxsKind::kParticleInelastic},
    {"ParticleInelastic", "triton", "triton", PxsKind::kParticleInelastic},
    {"ParticleInelastic", "He3", "He3", PxsKind::kParticleInelastic},
    {"ParticleInelastic", "alpha", "alpha", PxsKind::kParticleInelastic},
    {"NeutronInelastic", "neutron", "neutron", PxsKind::kNeutronInelastic},
    {"NeutronElastic", "neutron", "neutron", PxsKind::kNeutronElastic},
    {"NeutronCapture", "neutron", "neutron", PxsKind::kNeutronCapture},
    {"GammaNuclear", "gamma", "gamma", PxsKind::kGammaNuclear},
};
constexpr int kNSets = 9;

Projectile<real_t> projectile_for(const char* name) {
  if (!std::strcmp(name, "proton")) { return proton<real_t>(); }
  if (!std::strcmp(name, "neutron")) { return neutron<real_t>(); }
  if (!std::strcmp(name, "deuteron")) { return deuteron<real_t>(); }
  if (!std::strcmp(name, "triton")) { return triton<real_t>(); }
  if (!std::strcmp(name, "He3")) { return he3<real_t>(); }
  if (!std::strcmp(name, "alpha")) { return alpha<real_t>(); }
  return gamma<real_t>();
}

/// The nine loaded data sets. The tables are held separately from the data sets because a
/// PxsDataSet holds a pointer into one; both must outlive every evaluation.
struct Loaded {
  data::ParticleXsTable<real_t> table[kNSets];
  PxsDataSet<real_t> ds[kNSets];
  bool ok[kNSets] = {};
};

int load_all(Loaded& L) {
  int fails = 0;
  for (int i = 0; i < kNSets; ++i) {
    const std::string dir = host::g4particlexs_subdir(kSets[i].dir);
    if (dir.empty()) {
      std::printf("FAIL: no G4PARTICLEXS dataset found - g4particlexs_dir() is empty. Set "
                  "G4PARTICLEXSDATA or G4GPU_DATA_DIR.\n");
      return 1;
    }
    L.ok[i] = pxs_load<real_t>(kSets[i].kind, projectile_for(kSets[i].particle), dir,
                               L.table[i], L.ds[i]);
    if (!L.ok[i]) {
      std::printf("FAIL: could not load %s from %s\n", kSets[i].dataset, dir.c_str());
      ++fails;
    }
  }
  return fails;
}

// ------------------------------------------------------------------ 1. element cross sections

int check_element(const std::string& dir, Loaded& L, Cell* cells) {
  FILE* f = open_oracle(dir, "had_particlexs.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char dset[40], pname[32];
    int z = 0;
    double e = 0, xs = 0;
    if (std::sscanf(line, "%39[^,],%31[^,],%d,%lf,%lf", dset, pname, &z, &e, &xs) != 5) {
      continue;
    }
    int k = -1;
    for (int i = 0; i < kNSets; ++i) {
      if (!std::strcmp(dset, kSets[i].dataset) && !std::strcmp(pname, kSets[i].particle)) {
        k = i;
        break;
      }
    }
    if (k < 0 || !L.ok[k]) { continue; }
    // `loge` is what the caller of the real class supplies:
    // G4DynamicParticle::GetLogKineticEnergy(), i.e. G4Log(ekin), which on Windows is
    // std::log. Computed here rather than inside the port so that the port stays
    // device-callable with a precomputed logarithm, as Geant4's own callers pass one.
    const XsValue<real_t> v = pxs_element_xs<real_t>(L.ds[k], e, std::log(e), z);
    if (!v.ok()) {
      ++cells[k].refused;
      continue;
    }
    cmp(cells[k], v.value, xs, "%s %s Z=%d at %.17g MeV", dset, pname, z, e);
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------ 2. isotope cross sections

int check_isotope(const std::string& dir, Loaded& L, Cell* cells) {
  FILE* f = open_oracle(dir, "had_particlexs_iso.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char dset[40], pname[32];
    int z = 0, a = 0;
    double e = 0, xs = 0;
    if (std::sscanf(line, "%39[^,],%31[^,],%d,%d,%lf,%lf", dset, pname, &z, &a, &e, &xs) != 6) {
      continue;
    }
    int k = -1;
    for (int i = 0; i < kNSets; ++i) {
      if (!std::strcmp(dset, kSets[i].dataset) && !std::strcmp(pname, kSets[i].particle)) {
        k = i;
        break;
      }
    }
    if (k < 0 || !L.ok[k]) { continue; }
    const XsValue<real_t> v = pxs_iso_xs<real_t>(L.ds[k], e, std::log(e), z, a);
    if (!v.ok()) {
      ++cells[k].refused;
      continue;
    }
    cmp(cells[k], v.value, xs, "%s %s Z=%d A=%d at %.17g MeV", dset, pname, z, a, e);
  }
  std::fclose(f);
  return 0;
}

// ------------------------------------------------------------------ 3. the neutron general
//                                                                   process's own table

/// The materials had_matelem.csv describes, in the order they first appear - which is
/// G4Material::GetIndex() order, the same order the stored tables are in.
struct MatSet {
  std::vector<std::string> names;
  std::vector<data::Material<real_t>> mats;
  int index_of(const std::string& n) const {
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == n) { return static_cast<int>(i); }
    }
    return -1;
  }
};

int read_matelem(const std::string& dir, MatSet& out) {
  FILE* f = open_oracle(dir, "had_matelem.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char name[128];
    int idx = 0, z = 0;
    double na = 0;
    if (std::sscanf(line, "%127[^,],%d,%d,%lf", name, &idx, &z, &na) != 4) { continue; }
    int m = out.index_of(name);
    if (m < 0) {
      out.names.push_back(name);
      out.mats.push_back(data::Material<real_t>{});
      m = static_cast<int>(out.names.size()) - 1;
      out.mats[m].n_elements = 0;
    }
    data::Material<real_t>& mat = out.mats[m];
    if (idx != mat.n_elements) {
      std::printf("FAIL: had_matelem.csv has %s element index %d where %d was expected - the "
                  "element order is what the atom densities are keyed by\n", name, idx,
                  mat.n_elements);
      std::fclose(f);
      return 1;
    }
    if (mat.n_elements >= data::kMaxElements) {
      std::printf("FAIL: %s has more than kMaxElements=%d elements\n", name,
                  data::kMaxElements);
      std::fclose(f);
      return 1;
    }
    mat.z[mat.n_elements] = static_cast<real_t>(z);
    mat.n_atoms[mat.n_elements] = static_cast<real_t>(na);
    ++mat.n_elements;
  }
  std::fclose(f);
  return 0;
}

int check_neutron_general(const std::string& dir, Loaded& L, const MatSet& ms, Cell* cells,
                          NeutronGeneralTable<real_t>& ngt) {
  // The three data sets G4NeutronGeneralProcess takes are the first of each sub-process's
  // store: G4NeutronElasticXS, G4NeutronInelasticXS and G4NeutronCaptureXS.
  const int iel = 6, iin = 5, icap = 7;
  if (!L.ok[iel] || !L.ok[iin] || !L.ok[icap]) { return 1; }
  ngp_build_table<real_t>(L.ds[iel], L.ds[iin], L.ds[icap], ms.mats.data(),
                          static_cast<int>(ms.mats.size()), ngt);

  FILE* f = open_oracle(dir, "had_neutron_general.csv");
  if (f == nullptr) { return 1; }
  char line[1024];
  int unknown_material = 0;
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char name[128];
    int it = 0;
    long j = 0;
    double e = 0, v = 0;
    if (std::sscanf(line, "%127[^,],%d,%ld,%lf,%lf", name, &it, &j, &e, &v) != 5) { continue; }
    const int m = ms.index_of(name);
    if (m < 0) {
      ++unknown_material;
      continue;
    }
    if (it < 0 || it > 4) { continue; }
    const bool low = (it <= 2);
    const std::vector<real_t>& grid = low ? ngt.e_low : ngt.e_high;
    const int nn = low ? ngt.n_low : ngt.n_high;
    if (j < 0 || j >= nn) {
      std::printf("FAIL: table %d node %ld is outside the port's %d nodes for %s - the grid "
                  "itself is wrong\n", it, j, nn, name);
      std::fclose(f);
      return 1;
    }
    // The GRID first, then the value. A wrong node count or a wrong bin width leaves the two
    // ends exact and every interior node in the wrong place, which is the failure docs/RISK.md
    // V5 describes; comparing only values at Geant4's energies would not see it, because the
    // energies would be Geant4's.
    cmp(cells[5], static_cast<double>(grid[static_cast<std::size_t>(j)]), e,
        "grid table=%d node=%ld %s", it, j, name);
    const std::vector<real_t>* tv = nullptr;
    switch (it) {
      case 0: tv = &ngt.t0; break;
      case 1: tv = &ngt.t1; break;
      case 2: tv = &ngt.t2; break;
      case 3: tv = &ngt.t3; break;
      default: tv = &ngt.t4; break;
    }
    const std::size_t off = static_cast<std::size_t>(m * nn + j);
    cmp(cells[it], static_cast<double>((*tv)[off]), v, "table=%d node=%ld %s at %.17g MeV", it,
        j, name, e);
  }
  std::fclose(f);
  if (unknown_material != 0) {
    std::printf("FAIL: %d rows of had_neutron_general.csv name a material had_matelem.csv "
                "does not describe - the table cannot be rebuilt for it\n", unknown_material);
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  const char* env = std::getenv("G4GPU_ORACLE");
  const std::string dir = (env != nullptr) ? env : "ref/oracle";

  static Loaded L;
  int fails = load_all(L);
  if (fails != 0) {
    std::printf("\nFAILED (%d data sets did not load)\n", fails);
    return 1;
  }
  std::printf("== G4PARTICLEXS data sets vs Geant4 11.1.1 ==\n");
  std::printf("   dataset directory: %s\n\n", host::g4particlexs_dir().c_str());

  Cell elem[kNSets], iso[kNSets], ngen[6];
  int io = 0;
  io += check_element(dir, L, elem);
  io += check_isotope(dir, L, iso);
  MatSet ms;
  io += read_matelem(dir, ms);
  static NeutronGeneralTable<real_t> ngt;
  if (io == 0) { io += check_neutron_general(dir, L, ms, ngen, ngt); }
  if (io != 0) {
    std::printf("\nFAILED (%d oracle problems)\n", io);
    return 1;
  }

  auto report = [&](const char* group, const char* name, const Cell& c, bool may_be_empty) {
    std::printf("  %-18s %-24s %8d points", group, name, c.n);
    if (c.refused > 0) {
      std::printf("  %7d refused", c.refused);
    } else {
      std::printf("%16s", "");
    }
    std::printf("  worst %10.3e\n", c.worst);
    if (c.n == 0 && !may_be_empty) {
      std::printf("    FAIL: no points compared for %s / %s\n", group, name);
      ++fails;
    }
    if (c.worst > kTol) {
      std::printf("    FAIL: %s / %s off by %.3e, limit %.0e\n      %s\n", group, name, c.worst,
                  kTol, c.where.c_str());
      ++fails;
    }
  };

  for (int i = 0; i < kNSets; ++i) {
    char n[80];
    std::snprintf(n, sizeof n, "%s %s", kSets[i].dataset, kSets[i].particle);
    report("element", n, elem[i], false);
  }
  for (int i = 0; i < kNSets; ++i) {
    char n[80];
    std::snprintf(n, sizeof n, "%s %s", kSets[i].dataset, kSets[i].particle);
    // G4NeutronElasticXS has no isotope files at all, but ComputeIsoCrossSection still
    // answers - element cross section times A/aeff[Z] - so its bucket must not be empty
    // either. Only the gamma set may be, and only because it refuses above its tables.
    report("isotope", n, iso[i], false);
  }
  const char* tn[6] = {"table 0 (el+inel+cap)", "table 1 (el/sum)", "table 2 ((el+inel)/sum)",
                       "table 3 (el+inel)",     "table 4 (inel/sum)", "grid nodes"};
  for (int i = 0; i < 6; ++i) { report("NeutronGeneral", tn[i], ngen[i], false); }

  // ---------------------------------------------------------------- the grid, structurally
  //
  // nLowE = 100*G4lrint(log10(20 MeV / 1 keV)) = 100*lrint(4.301) = 400 bins, so 401 nodes;
  // nHighE = 10*G4lrint(log10(100 TeV / 20 MeV)) = 10*lrint(6.699) = 70 bins, 71 nodes.
  // Reading the 100 and the 10 as bins-per-decade would give 431 and 67 - both ends still
  // exact, every interior node moved. Asserted on the numbers and not only through the CSV so
  // that it fails even if the oracle is regenerated by a dumper that lost the grid column.
  if (ngt.n_low != 401 || ngt.n_high != 71) {
    std::printf("  FAIL: the general process's grid is %d + %d nodes, expected 401 + 71\n",
                ngt.n_low, ngt.n_high);
    ++fails;
  }
  if (ngp_n_low_bins() != 400 || ngp_n_high_bins() != 70) {
    std::printf("  FAIL: nLowE = %d and nHighE = %d, expected 400 and 70 - G4lrint rounds "
                "log10(20000) = 4.301 to 4, not up\n", ngp_n_low_bins(), ngp_n_high_bins());
    ++fails;
  }

  // ---------------------------------------------------------------- SampleZandA's other half
  //
  // G4CrossSectionDataStore::SampleZandA does not recompute anything: it reads the running
  // partial sums ComputeCrossSection left behind. So what can be checked against Geant4's own
  // numbers is that half - and it can be checked exactly, because
  // G4NeutronGeneralProcess's table 0 IS the sum of the three data sets' macroscopic cross
  // sections over the same material and the same elements. If store_compute_cross_section
  // agrees with it for all three sets summed, then the partial sums SampleZandA draws from are
  // right; what remains is the draw itself, and that is asserted structurally below.
  {
    Cell store;
    Cell draw;
    const int iel = 6, iin = 5, icap = 7;
    for (std::size_t m = 0; m < ms.mats.size(); ++m) {
      // Natural-abundance elements, which is what a NIST material has and what makes
      // G4CrossSectionDataStore::GetCrossSection take its element-wise branch.
      ElementIsotopes<real_t> isos[data::kMaxElements];
      for (int i = 0; i < ms.mats[m].n_elements; ++i) { isos[i].natural_abundance = true; }
      // Every twentieth node of the low grid, so the check spans the whole range without
      // re-walking 401 x 7 x 3 element sums.
      for (int j = 0; j < ngt.n_low; j += 20) {
        const real_t e = ngt.e_low[static_cast<std::size_t>(j)];
        const real_t le = std::log(e);
        MaterialXs<real_t> mx[3];
        real_t sum = 0;
        const int ks[3] = {iel, iin, icap};
        for (int q = 0; q < 3; ++q) {
          const XsValue<real_t> t =
              store_compute_cross_section<real_t>(L.ds[ks[q]], e, le, ms.mats[m], isos, mx[q]);
          if (!t.ok()) {
            ++store.refused;
            sum = -1;
            break;
          }
          sum += t.value;
        }
        if (sum < 0) { continue; }
        cmp(store, static_cast<double>(sum),
            static_cast<double>(ngt.t0[static_cast<std::size_t>(
                static_cast<int>(m) * ngt.n_low + j)]),
            "ComputeCrossSection sum %s at %.17g MeV", ms.names[m].c_str(),
            static_cast<double>(e));

        // The draw. Two invariants that a wrong cumulative array breaks and a per-element
        // one would not: the last cumulative entry is the total, and a draw of q selects the
        // element whose cumulative interval contains q*total. Checked at both ends and in the
        // middle, for the inelastic set.
        const MaterialXs<real_t>& mxi = mx[1];
        if (mxi.n_elements > 0 && mxi.total > 0) {
          cmp(draw, static_cast<double>(mxi.cumulative[mxi.n_elements - 1]),
              static_cast<double>(mxi.total), "cumulative[last] == total for %s at %.17g MeV",
              ms.names[m].c_str(), static_cast<double>(e));
          for (int i = 1; i < mxi.n_elements; ++i) {
            if (mxi.cumulative[i] < mxi.cumulative[i - 1]) {
              std::printf("  FAIL: %s cumulative partial sums are not monotonic at element "
                          "%d\n", ms.names[m].c_str(), i);
              ++fails;
            }
          }
          const real_t qs[5] = {real_t(0), real_t(0.25), real_t(0.5), real_t(0.75), real_t(1)};
          for (real_t q : qs) {
            const TargetZA t = store_sample_za<real_t>(L.ds[iin], e, le, ms.mats[m], isos, mxi,
                                                       q, real_t(0.5));
            const real_t cross = mxi.total * q;
            int want = 0;
            for (int i = 0; i < mxi.n_elements; ++i) {
              if (cross <= mxi.cumulative[i]) {
                want = i;
                break;
              }
            }
            // A single-element material never draws, as Geant4 does not either
            // (`if(1 < nElements)`), so element 0 is the answer whatever q is.
            if (mxi.n_elements == 1) { want = 0; }
            if (t.element_index != want ||
                t.z != static_cast<int>(ms.mats[m].z[want])) {
              std::printf("  FAIL: SampleZandA(%s, %.17g MeV, q=%.2f) chose element %d (Z=%d), "
                          "expected %d (Z=%d)\n", ms.names[m].c_str(), static_cast<double>(e),
                          static_cast<double>(q), t.element_index, t.z, want,
                          static_cast<int>(ms.mats[m].z[want]));
              ++fails;
            }
          }
        }
      }
    }
    report("SampleZandA", "ComputeCrossSection sum", store, false);
    report("SampleZandA", "cumulative[last]==total", draw, false);
  }

  // ---------------------------------------------------------------- G4PhysicsVectorType
  //
  // The enum must be Geant4's, because a value read from a Geant4 dump or a stored
  // G4PhysicsVector would be Geant4's. It was free=0, log=1, linear=2 for a while, under a
  // comment saying it was Geant4's encoding, and nothing failed because the port both wrote
  // and read it. Asserted on the numbers so the comment cannot drift from them again.
  if (kFreeVector != 0 || kLinearVector != 1 || kLogVector != 2) {
    std::printf("  FAIL: PhysVecType is free=%d linear=%d log=%d; G4PhysicsVectorType.hh is "
                "T_G4PhysicsFreeVector = 0, T_G4PhysicsLinearVector, T_G4PhysicsLogVector\n",
                static_cast<int>(kFreeVector), static_cast<int>(kLinearVector),
                static_cast<int>(kLogVector));
    ++fails;
  }
  // And the types the loader actually assigned, since the two wrongs cancelled last time: the
  // four hadronic sets are log vectors throughout, gamma's element files are linear except for
  // freeVectorException, and every gamma isotope file is free.
  {
    const int hadronic_sets[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int q = 0; q < 8; ++q) {
      const int k = hadronic_sets[q];
      for (int z = 1; z <= 92; ++z) {
        if (L.table[k].element[z].n > 0 && L.table[k].element[z].type != kLogVector) {
          std::printf("  FAIL: %s %s element Z=%d is vector type %d, expected kLogVector\n",
                      kSets[k].dataset, kSets[k].particle, z, L.table[k].element[z].type);
          ++fails;
          break;
        }
      }
    }
    const int kg = 8;  // GammaNuclear
    for (int z = 1; z <= 94; ++z) {
      const int want = data::pxs_gamma_free_vector_exception(z) ? kFreeVector : kLinearVector;
      if (L.table[kg].element[z].n > 0 && L.table[kg].element[z].type != want) {
        std::printf("  FAIL: gamma element Z=%d is vector type %d, expected %d "
                    "(freeVectorException = {4,6,7,8,27,39,45,65,67,69,73})\n", z,
                    L.table[kg].element[z].type, want);
        ++fails;
        break;
      }
    }
  }

  // ---------------------------------------------------------------- Z > 94 has no aeff
  //
  // G4NeutronElasticXS::ComputeIsoCrossSection divides by aeff[Z] at the UNCLAMPED Z, so for
  // Z >= 95 Geant4 reads past the end of a 95-entry array. That is undefined behaviour and not
  // a value to reproduce; the port must refuse rather than divide by the zero its accessor
  // returns, which would be an inf handed to a caller as a cross section.
  {
    const XsValue<real_t> v =
        pxs_iso_xs<real_t>(L.ds[6], real_t(1.0), std::log(real_t(1.0)), 95, 240);
    if (v.ok()) {
      std::printf("  FAIL: NeutronElastic isotope cross section at Z=95 answered %.17g instead "
                  "of refusing - aeff[] has 95 entries and Geant4 reads past them\n",
                  static_cast<double>(v.value));
      ++fails;
    } else if (v.refused != XsRefusal::kIsotopeListOutOfRange) {
      std::printf("  FAIL: Z=95 refused with %s, expected the aeff range refusal\n",
                  xs_refusal_name(v.refused));
      ++fails;
    }
  }

  // ---------------------------------------------------------------- two bad dataset files
  //
  // Built here rather than asserted in prose, and TWO of them, because Geant4 treats the two
  // failures differently and the port used to treat them the same:
  //
  //   * an element file that will not OPEN     -> FatalException (warn = true)
  //   * ANY file that opens and will not PARSE -> FatalException, unconditionally
  //
  // `warn` is false for an isotope file, so a missing isotope file is a legitimate null vector
  // the readers fall back from - but a malformed one is not. The port returned one `false` for
  // both and turned a corrupt isotope file into the element cross section times A/aeff[Z]: a
  // plausible number, from a file Geant4 would have aborted on.
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "g4gpu_pxs_bad_test";
    const fs::path src = fs::path(host::g4particlexs_dir()) / "neutron";

    // (a) el1 present, el2 deliberately absent. The loader walks Z upward and must stop at 2.
    fs::remove_all(root, ec);
    fs::create_directories(root / "neutron", ec);
    fs::copy_file(src / "el1", root / "neutron" / "el1", fs::copy_options::overwrite_existing,
                  ec);
    if (ec) {
      std::printf("  FAIL: could not stage the incomplete dataset (%s)\n", ec.message().c_str());
      ++fails;
    } else {
      data::ParticleXsTable<real_t> t;
      PxsDataSet<real_t> ds;
      std::printf("\n  -- expect a FATAL line below: element file absent --\n");
      if (pxs_load<real_t>(PxsKind::kNeutronElastic, neutron<real_t>(),
                           (root / "neutron").string(), t, ds)) {
        std::printf("  FAIL: pxs_load accepted a dataset directory with neutron/el2 missing - "
                    "a missing element file must be fatal, never a zero cross section\n");
        ++fails;
      } else {
        std::printf("  -- refused, as it must --\n");
      }
    }

    // (b) a complete CAPTURE dataset with ONE isotope file replaced by garbage. Capture is used
    // rather than elastic because G4NeutronElasticXS opens no isotope file at all, so the
    // branch under test would never run.
    fs::remove_all(root, ec);
    fs::create_directories(root / "cap", ec);
    const fs::path csrc = fs::path(host::g4particlexs_dir()) / "neutron";
    bool staged = true;
    for (int z = 1; z <= 92 && staged; ++z) {
      const std::string cn = "cap" + std::to_string(z);
      fs::copy_file(csrc / cn, root / "cap" / cn, fs::copy_options::overwrite_existing, ec);
      if (ec) { staged = false; }
      for (int a = data::isotope_amin()[z]; a <= data::isotope_amax()[z]; ++a) {
        const std::string in = cn + "_" + std::to_string(a);
        std::error_code ec2;
        fs::copy_file(csrc / in, root / "cap" / in, fs::copy_options::overwrite_existing, ec2);
      }
    }
    if (!staged) {
      std::printf("  FAIL: could not stage the corrupt-isotope dataset (%s)\n",
                  ec.message().c_str());
      ++fails;
    } else {
      // cap1_2 exists in G4PARTICLEXS4.0; overwrite it with a header that opens and does not
      // parse - a node count of 1, which G4PhysicsVector::Retrieve rejects.
      FILE* bad = std::fopen((root / "cap" / "cap1_2").string().c_str(), "w");
      if (bad == nullptr) {
        std::printf("  FAIL: could not write the corrupt isotope file\n");
        ++fails;
      } else {
        std::fprintf(bad, "1e-06 20 1\n1\n1e-06 3.2e-25\n");
        std::fclose(bad);
        data::ParticleXsTable<real_t> t;
        PxsDataSet<real_t> ds;
        std::printf("\n  -- expect a FATAL line below: isotope file opens and does not parse "
                    "--\n");
        if (pxs_load<real_t>(PxsKind::kNeutronCapture, neutron<real_t>(),
                             (root / "cap").string(), t, ds)) {
          std::printf("  FAIL: pxs_load accepted a dataset whose neutron/cap1_2 opens and does "
                      "not parse. Geant4's Retrieve failure is a FatalException and is NOT "
                      "guarded by `warn` - only the open failure is. A corrupt isotope file "
                      "must not become the element cross section times A/aeff[Z].\n");
          ++fails;
        } else {
          std::printf("  -- refused, as it must --\n");
        }
      }
    }
    fs::remove_all(root, ec);
  }

  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
