// The three arrays G4IsotopeList.hh holds, and every G4PARTICLEXS reader indexes by Z.
//
// Transcribed verbatim from source/processes/hadronic/cross_sections/include/G4IsotopeList.hh
// (11.1.1): `amin[]`, `amax[]`, `aeff[]`, Z = 0..94.
//
// They are not a convenience. Each decides a branch:
//
//   amin/amax   the mass-number window a per-isotope data file exists for. Every reader tests
//               `amin[Z] < amax[Z] && A >= amin[Z] && A <= amax[Z]` before it looks for one,
//               and the *index* into the per-isotope block is `A - amin[Z]`. Where
//               amin == amax there is no isotope data at all and the reader falls back to the
//               element vector times A/aeff[Z] - a different number, not a rounding of it.
//   aeff        the mean mass number the element cross section is *defined for*. It is not
//               data/materials.cuh's `atomic_mass` (G4NistManager::GetAtomicMassAmu) and not
//               G4Element::GetA(): 12.0107 here against 12.010736 there for carbon. The
//               isotope cross section is the element one scaled by A/aeff[Z], and the
//               Glauber-Gribov hand-over above the table is evaluated at A = aeff[Z], so
//               substituting the more accurate number changes both by ~3e-6 - which is
//               3e6 times the tolerance the oracle is compared at.
//
// Note the ceiling differs from the readers' own: MAXZINEL, MAXZINELP, MAXZEL and MAXZCAPTURE
// are all 93 (so Z is clamped to 92), MAXZGAMMAXS is 95, and these arrays run to Z = 94. A
// reader clamping Z to 92 never reads index 93 or 94; G4GammaNuclearXS can reach 94.
#ifndef G4GPU_DATA_ISOTOPE_LIST_HH
#define G4GPU_DATA_ISOTOPE_LIST_HH

namespace g4gpu::data {

/// Highest Z the three arrays below are defined for.
constexpr int kIsotopeListMaxZ = 94;

/// G4IsotopeList.hh `amin` - lowest A with a per-isotope data file, indexed by Z.
__host__ __device__ inline const int* isotope_amin() {
  static const int v[kIsotopeListMaxZ + 1] = {
    0, 1, 3, 6, 9, 10, 12, 14, 16, 19,
    20, 23, 24, 27, 27, 31, 32, 35, 36, 39,
    40, 45, 46, 50, 50, 55, 54, 59, 58, 63,
    64, 69, 70, 75, 74, 79, 78, 85, 84, 89,
    90, 93, 92, 98, 96, 103, 102, 107, 106, 113,
    112, 121, 120, 127, 124, 133, 130, 138, 136, 141,
    142, 145, 144, 151, 152, 158, 156, 165, 162, 169,
    168, 175, 174, 180, 180, 185, 184, 191, 190, 197,
    196, 203, 204, 209, 209, 210, 222, 223, 226, 227,
    232, 231, 233, 237, 238};
  return v;
}

/// G4IsotopeList.hh `amax` - highest A with a per-isotope data file, indexed by Z.
__host__ __device__ inline const int* isotope_amax() {
  static const int v[kIsotopeListMaxZ + 1] = {
    0, 3, 4, 7, 9, 11, 14, 15, 18, 19,
    22, 23, 26, 27, 30, 31, 36, 37, 40, 41,
    48, 45, 50, 51, 54, 55, 58, 59, 64, 65,
    70, 71, 76, 75, 82, 81, 86, 87, 90, 89,
    96, 94, 100, 98, 104, 103, 110, 109, 116, 115,
    124, 123, 130, 129, 136, 137, 138, 139, 142, 141,
    150, 145, 154, 153, 160, 159, 164, 165, 170, 169,
    176, 176, 180, 181, 186, 187, 192, 193, 198, 197,
    204, 205, 208, 209, 209, 210, 222, 223, 226, 227,
    232, 231, 238, 237, 244};
  return v;
}

/// G4IsotopeList.hh `aeff` - the mean A the element cross section is defined at.
__host__ __device__ inline const double* isotope_aeff() {
  static const double v[kIsotopeListMaxZ + 1] = {
    0., 1.00794, 4.00264, 6.94003, 9.01218, 10.811, 12.0107, 14.0068,
    15.9994, 18.9984, 20.18, 22.9898, 24.305, 26.9815, 28.0854, 30.9738,
    32.0661, 35.4526, 39.9477, 39.0983, 40.078, 44.9559, 47.8667, 50.9415,
    51.9961, 54.938, 55.8451, 58.9332, 58.6933, 63.5456, 65.3955, 69.7231,
    72.6128, 74.9216, 78.9594, 79.9035, 83.7993, 85.4677, 87.6166, 88.9058,
    91.2236, 92.9064, 95.9313, 97.9072, 101.065, 102.906, 106.415, 107.868,
    112.411, 114.818, 118.71, 121.76, 127.603, 126.904, 131.292, 132.905,
    137.327, 138.905, 140.115, 140.908, 144.236, 144.913, 150.366, 151.964,
    157.252, 158.925, 162.497, 164.93, 167.256, 168.934, 173.038, 174.967,
    178.485, 180.948, 183.842, 186.207, 190.225, 192.216, 195.078, 196.967,
    200.599, 204.383, 207.217, 208.98, 208.982, 209.987, 222.018, 223.02,
    226.025, 227.028, 232.038, 231.036, 238.029, 237.048, 244.064};
  return v;
}

template <typename real_t>
__host__ __device__ inline real_t isotope_aeff_of(int z) {
  return (z >= 0 && z <= kIsotopeListMaxZ) ? static_cast<real_t>(isotope_aeff()[z])
                                           : real_t(0);
}

}  // namespace g4gpu::data

#endif  // G4GPU_DATA_ISOTOPE_LIST_HH
