// The G4PARTICLEXS reader: per-element and per-isotope cross-section vectors, read on the
// host into one flat table the device indexes into.
//
// Format, from G4PhysicsVector::Retrieve(fIn, ascii=true) - which is what every
// G4PARTICLEXS reader calls and the only definition of the format there is:
//
//     edgeMin edgeMax numberOfNodes
//     numberOfNodes
//     E_0 value_0
//     ...
//     E_{n-1} value_{n-1}
//
// with `numberOfNodes < 2` and a second count that disagrees with the first both being
// failures. The first two numbers are read and then **overwritten** by Initialise(), which
// takes edgeMin and edgeMax from binVector[0] and binVector[n-1]; they are only a header, and
// they are not always right - gamma/inel1 declares `0 130 2`, and 0 is indeed its first node,
// but nothing depends on the declaration. The values are already in Geant4 internal units -
// MeV and mm^2, no ScaleVector call anywhere - so nothing is multiplied on the way in. n+O16
// elastic at 1 keV reads 3.856e-22, which is 3.856 barn, which is the number.
//
// FILE LAYOUT ON DISK
//
//     <G4PARTICLEXSDATA>/proton/inel<Z>            element,  Z = 1..92
//     <G4PARTICLEXSDATA>/proton/inel<Z>_<A>        isotope,  amin[Z] <= A <= amax[Z]
//     <G4PARTICLEXSDATA>/neutron/{el,inel,cap}<Z> and <Z>_<A>
//     <G4PARTICLEXSDATA>/{deuteron,triton,He3,alpha}/inel<Z> and <Z>_<A>
//     <G4PARTICLEXSDATA>/gamma/inel<Z> and <Z>_<A>, Z = 1..94
//
// An element file that will not open is a FatalException in Geant4 (`warn = true`); an isotope
// file that will not open is silently skipped, and the reader falls back to the element vector
// scaled by A/aeff[Z]. Both are reproduced: a missing element file makes `load_particlexs`
// return false, which every caller treats as fatal, and a missing isotope file leaves that
// slot with zero length, which the evaluation reads as "no isotope data" exactly as Geant4
// reads a null pointer. Not every A in [amin, amax] has a file - Geant4 asks for all of them
// and accepts the misses, so the absent ones are part of the format.
//
// WHICH VECTOR TYPE, PER DATASET
//
// It is not uniform, and the type decides the bin lookup:
//
//   proton/inel, neutron/el, neutron/inel, neutron/cap, {d,t,He3,alpha}/inel, and every
//   isotope file of those      G4PhysicsLogVector   (`new G4PhysicsLogVector()`, spline off)
//   gamma/inel<Z>, Z not in
//   {4,6,7,8,27,39,45,65,67,69,73}   G4PhysicsLinearVector
//   gamma/inel<Z>, Z in that list,
//   and every gamma isotope file     G4PhysicsVector (free)
//
// The gamma exception list is `freeVectorException` in G4GammaNuclearXS.hh. Its members are
// the elements whose IAEA data are on an irregular grid - oxygen's runs 11.499, 11.5, 11.604,
// 12.132 - where the rest are on a uniform 0.5 MeV one. Reading an irregular grid as a linear
// vector would compute the bin from `(E - edgeMin)*invdBin` and land in the wrong place:
// right at the nodes, wrong between them, which is the shape of error a coarse test does not
// see.
//
// The vector's derived parameters (edgeMin, edgeMax, invdBin, logemin) are computed once here,
// at load time, and carried in the slice - so evaluation needs no logarithm of a table edge
// and is callable from device code.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "data/isotope_list.hh"

namespace g4gpu::data {

/// The highest Z any G4PARTICLEXS reader indexes: MAXZGAMMAXS is 95, the other four are 93.
constexpr int kPxsMaxZ = 95;

/// One vector's slice of the flat (energy, value) arrays, with the Initialise() results.
///
/// `n == 0` means the file was absent, which for an isotope is legitimate and for an element
/// is fatal. `type` is g4gpu::hadronic::xs::PhysVecType: 0 free, 1 log, 2 linear.
template <typename real_t>
struct PxsSlice {
  int off = 0;
  int n = 0;
  int type = 0;
  real_t edge_min = 0;
  real_t edge_max = 0;
  real_t inv_dbin = 0;
  real_t log_emin = 0;
};

/// One dataset - one particle and one channel - as the readers hold it.
///
/// Isotope slices are stored contiguously per Z, `iso_base[Z] .. iso_base[Z]+iso_n[Z]-1`,
/// indexed by `A - amin[Z]`, which is exactly G4ElementData::GetComponentDataByIndex's
/// contract.
template <typename real_t>
struct ParticleXsTable {
  PxsSlice<real_t> element[kPxsMaxZ];
  int iso_base[kPxsMaxZ] = {0};  ///< -1 when this Z has no isotope block at all
  int iso_n[kPxsMaxZ] = {0};     ///< amax[Z] - amin[Z] + 1, or 0

  std::vector<PxsSlice<real_t>> isotopes;
  std::vector<real_t> table_e;  ///< concatenated energies, MeV
  std::vector<real_t> table_v;  ///< concatenated values, mm^2

  // The three pointers the evaluation reads, set once at the end of load_particlexs.
  //
  // They exist because the evaluation is __host__ __device__ and std::vector::data() is a
  // host-only member function - calling it from a __host__ __device__ function is an nvcc
  // error, not a warning. A device build replaces these three with device allocations and
  // leaves the vectors empty; nothing else in the struct has to change.
  const real_t* e_data = nullptr;
  const real_t* v_data = nullptr;
  const PxsSlice<real_t>* iso_data = nullptr;

  int max_z = 0;  ///< highest Z actually loaded, inclusive
};

/// G4GammaNuclearXS::freeVectorException - the eleven elements whose gamma element file is
/// read as a free vector rather than a linear one.
inline bool pxs_gamma_free_vector_exception(int z) {
  static const int v[11] = {4, 6, 7, 8, 27, 39, 45, 65, 67, 69, 73};
  for (int i = 0; i < 11; ++i) {
    if (v[i] == z) { return true; }
  }
  return false;
}

/// The Initialise() of whichever vector type the slice is, run once at load time.
template <typename real_t>
inline void pxs_initialise_slice(PxsSlice<real_t>& s, const std::vector<real_t>& es) {
  if (s.n < 2) { return; }
  s.edge_min = es[static_cast<std::size_t>(s.off)];
  s.edge_max = es[static_cast<std::size_t>(s.off + s.n - 1)];
  const real_t idxmax_plus1 = static_cast<real_t>(s.n - 1);
  if (s.type == 1) {  // G4PhysicsLogVector::Initialise
    s.inv_dbin = idxmax_plus1 / std::log(s.edge_max / s.edge_min);
    s.log_emin = std::log(s.edge_min);
  } else if (s.type == 2) {  // G4PhysicsLinearVector::Initialise
    s.inv_dbin = idxmax_plus1 / (s.edge_max - s.edge_min);
  }
}

/// Reads one file in G4PhysicsVector::Retrieve's ascii format. Returns false if it will not
/// open or will not parse.
template <typename real_t>
inline bool pxs_read_file(const std::string& path, std::vector<real_t>& es,
                          std::vector<real_t>& vs, PxsSlice<real_t>& out) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return false; }
  double emin = 0, emax = 0;
  int nnodes = 0, siz = 0;
  if (std::fscanf(f, "%lf %lf %d", &emin, &emax, &nnodes) != 3 || nnodes < 2) {
    std::fclose(f);
    return false;
  }
  if (std::fscanf(f, "%d", &siz) != 1 || siz != nnodes) {
    std::fclose(f);
    return false;
  }
  const int off = static_cast<int>(es.size());
  for (int i = 0; i < siz; ++i) {
    double e = 0, v = 0;
    if (std::fscanf(f, "%lf %lf", &e, &v) != 2) {
      std::fclose(f);
      es.resize(static_cast<std::size_t>(off));
      vs.resize(static_cast<std::size_t>(off));
      return false;
    }
    es.push_back(static_cast<real_t>(e));
    vs.push_back(static_cast<real_t>(v));
  }
  std::fclose(f);
  out.off = off;
  out.n = siz;
  pxs_initialise_slice(out, es);
  return true;
}

/// Loads one dataset.
///
/// @param dir        the particle subdirectory, e.g. "<G4PARTICLEXSDATA>/neutron"
/// @param prefix     "inel", "el" or "cap"
/// @param zmax       load Z = 1 .. zmax inclusive (92 for the four nucleon/ion readers, 94
///                   for gamma, which is MAXZGAMMAXS - 1 and the highest file on disk)
/// @param is_gamma   apply the G4GammaNuclearXS vector-type rules instead of the log one
/// @param with_isotopes  also load the per-isotope files, as every reader but
///                       G4NeutronElasticXS does (its IsIsoApplicable returns false and it
///                       never opens one)
/// @return false when an *element* file is missing, which is Geant4's FatalException.
///
/// The slices point into `t.table_e` / `t.table_v`, so those two vectors must not be reserved
/// or copied out from under a slice; everything is filled before any slice is read.
template <typename real_t>
inline bool load_particlexs(const std::string& dir, const std::string& prefix, int zmax,
                            bool is_gamma, bool with_isotopes, ParticleXsTable<real_t>& t) {
  t.isotopes.clear();
  t.table_e.clear();
  t.table_v.clear();
  for (int z = 0; z < kPxsMaxZ; ++z) {
    t.element[z] = PxsSlice<real_t>{};
    t.iso_base[z] = -1;
    t.iso_n[z] = 0;
  }
  t.max_z = zmax;

  const int* amin = isotope_amin();
  const int* amax = isotope_amax();

  for (int z = 1; z <= zmax; ++z) {
    PxsSlice<real_t> el;
    el.type = is_gamma ? (pxs_gamma_free_vector_exception(z) ? 0 : 2) : 1;
    const std::string p = dir + "/" + prefix + std::to_string(z);
    if (!pxs_read_file<real_t>(p, t.table_e, t.table_v, el)) {
      std::printf("\nFATAL: G4PARTICLEXS element file <%s> is not opened.\n"
                  "  Geant4 raises a FatalException here (G4*XS::RetrieveVector with"
                  " warn=true).\n"
                  "  Missing data is fatal, never a silent zero - see src/host/g4data.cuh.\n",
                  p.c_str());
      return false;
    }
    t.element[z] = el;

    if (!with_isotopes || z > kIsotopeListMaxZ) { continue; }
    if (amin[z] >= amax[z]) { continue; }
    const int nmax = amax[z] - amin[z] + 1;
    t.iso_base[z] = static_cast<int>(t.isotopes.size());
    t.iso_n[z] = nmax;
    for (int a = amin[z]; a <= amax[z]; ++a) {
      PxsSlice<real_t> iso;
      // Isotope files are free vectors for gamma and log vectors otherwise; the element's
      // linear/free choice does not apply to them.
      iso.type = is_gamma ? 0 : 1;
      const std::string pi =
          dir + "/" + prefix + std::to_string(z) + "_" + std::to_string(a);
      if (!pxs_read_file<real_t>(pi, t.table_e, t.table_v, iso)) { iso = PxsSlice<real_t>{}; }
      t.isotopes.push_back(iso);
    }
  }
  t.e_data = t.table_e.data();
  t.v_data = t.table_v.data();
  t.iso_data = t.isotopes.data();
  return true;
}

/// The isotope slice for (Z, A), or a zero-length one when there is no file - which is what
/// G4ElementData::GetComponentDataByIndex returning nullptr means.
template <typename real_t>
__host__ __device__ inline PxsSlice<real_t> pxs_isotope(const ParticleXsTable<real_t>& t,
                                                        int z, int a) {
  if (z < 1 || z >= kPxsMaxZ || t.iso_base[z] < 0 || t.iso_data == nullptr) {
    return PxsSlice<real_t>{};
  }
  const int i = a - isotope_amin()[z];
  if (i < 0 || i >= t.iso_n[z]) { return PxsSlice<real_t>{}; }
  return t.iso_data[t.iso_base[z] + i];
}

}  // namespace g4gpu::data
