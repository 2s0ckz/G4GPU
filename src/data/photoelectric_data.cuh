// Livermore photoelectric data: G4EMLOW epics2017/phot, loaded on the host into flat
// device-resident tables.
//
// Transcribed from G4LivermorePhotoElectricModel. The cross section has four branches:
//
//   E >= paramHigh[0]   5th-order polynomial in 1/E, high-energy coefficients
//   E >= paramLow[0]    same form, low-energy coefficients
//   E >= paramHigh[1]   x^-3 times a tabulated cross section (pe-cs-Z.dat)
//   otherwise           x^-3 times a tabulated cross section (pe-le-cs-Z.dat)
//
// The two polynomial branches cover everything above about 5 keV and are transcribed exactly.
// The tabulated branches below that are interpolated linearly - and so is Geant4's, in this
// configuration. `ReadData` computes
//
//     G4bool spline = (param->LivermoreDataDir() == "livermore");
//
// and the default directory is `epics2017`, not `livermore`, so the vector is built unsplined
// and `FillSecondDerivatives` returns immediately. The below-K-shell vector says so in a
// comment of its own: "no spline for photoeffect total x-section below K-shell".
//
// This file used to claim the linear interpolation was an approximation of a spline Geant4
// used. It is not, and the measurement was the clue: docs/PHYSICS_PLAN.md records the
// tabulated branch agreeing with Geant4 to 0.0004%, which is not what a cubic spline replaced
// by a straight line looks like on a grid this coarse. A stated approximation that does not
// exist is as much a misstatement as an unstated one - it sends the next person to fix
// something that is already right.
//
// File formats:
//   pe-high-Z.dat / pe-low-Z.dat : "nShells nShells Eboundary", then nShells rows of
//                                  [binding_MeV, c1..c6 in barn]
//   pe-cs-Z.dat / pe-le-cs-Z.dat : "emin emax n", then "n", then n rows of "E value"
//   pe-cs-Z.dat / pe-le-cs-Z.dat : "emin emax n", then "n", then n rows of "E value"
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/units.cuh"
#include "core/vec3.cuh"

namespace g4gpu::data {

constexpr int kPeMaxShells = 12;      ///< Ca has 8; 12 leaves headroom
constexpr int kPeMaxElements = 16;
constexpr int kPeParamStride = kPeMaxShells * 7 + 1;

/// One element's photoelectric parameterization. Tabulated points live in shared arrays.
template <typename real_t>
struct PhotoElectricElement {
  int z;
  int n_shells;
  real_t param_high[kPeParamStride];  ///< [0] = boundary energy, then 7 per shell
  real_t param_low[kPeParamStride];
  int cs_offset, cs_count;            ///< pe-cs-Z.dat  (E >= paramHigh[1])
  int le_offset, le_count;            ///< pe-le-cs-Z.dat
  /// Per-shell tabulated cross sections from pe-ss-cs-Z.dat, used for shell selection
  /// below both parameterizations. Stored in file order; ss_id is Geant4's component id.
  int n_ss;
  int ss_offset[kPeMaxShells], ss_count[kPeMaxShells], ss_id[kPeMaxShells];
};

/// Flat table for all elements the geometry uses.
template <typename real_t>
struct PhotoElectricTable {
  PhotoElectricElement<real_t> elements[kPeMaxElements];
  int n_elements;
  int z_to_index[101];  ///< -1 when the element is not loaded

  const real_t* table_e;  ///< concatenated energies, MeV
  const real_t* table_v;  ///< concatenated values, barn
  int table_n;
};

// ---------------------------------------------------------------- host loading

/// Reads "nShells nShells Eboundary" then nShells rows of 7. Values are scaled to Geant4
/// internal units exactly as ReadData does: column 0 by MeV, the rest by barn.
template <typename real_t>
__host__ inline bool load_param_file(const std::string& path, real_t* out, int& n_shells) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { return false; }
  int n1 = 0, n2 = 0;
  double x = 0;
  if (std::fscanf(f, "%d %d %lf", &n1, &n2, &x) != 3) { std::fclose(f); return false; }
  if (n1 < 1 || n1 > kPeMaxShells) { std::fclose(f); return false; }
  n_shells = n1;
  out[0] = real_t(x);  // already MeV
  int k = 1;
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < 7; ++j) {
      double v = 0;
      if (std::fscanf(f, "%lf", &v) != 1) { std::fclose(f); return false; }
      // j == 0 is a binding energy in MeV; the rest are barn.
      out[k++] = (j == 0) ? real_t(v) : real_t(v) * units::barn<real_t>();
    }
  }
  std::fclose(f);
  return true;
}

/// Reads "emin emax n", then "n", then n rows of "E value". Energies stay in MeV; values are
/// multiplied by @p vscale. Photoelectric files are in barn (ScaleVector(MeV, barn)),
/// Rayleigh files are already in Geant4 internal units, so that caller passes 1.
template <typename real_t>
__host__ inline bool load_cs_file_scaled(const std::string& path, std::vector<real_t>& es,
                                         std::vector<real_t>& vs, int& offset, int& count,
                                         real_t vscale) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { offset = 0; count = 0; return false; }
  double emin = 0, emax = 0;
  int n = 0, n_again = 0;
  if (std::fscanf(f, "%lf %lf %d", &emin, &emax, &n) != 3) { std::fclose(f); return false; }
  if (std::fscanf(f, "%d", &n_again) != 1) { std::fclose(f); return false; }
  offset = static_cast<int>(es.size());
  count = 0;
  for (int i = 0; i < n_again; ++i) {
    double e = 0, v = 0;
    if (std::fscanf(f, "%lf %lf", &e, &v) != 2) { break; }
    es.push_back(real_t(e));
    vs.push_back(real_t(v) * vscale);
    ++count;
  }
  std::fclose(f);
  return count > 0;
}

/// Photoelectric variant: values are in barn.
template <typename real_t>
__host__ inline bool load_cs_file(const std::string& path, std::vector<real_t>& es,
                                  std::vector<real_t>& vs, int& offset, int& count) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) { offset = 0; count = 0; return false; }
  double emin = 0, emax = 0;
  int n = 0, n_again = 0;
  if (std::fscanf(f, "%lf %lf %d", &emin, &emax, &n) != 3) { std::fclose(f); return false; }
  if (std::fscanf(f, "%d", &n_again) != 1) { std::fclose(f); return false; }
  offset = static_cast<int>(es.size());
  count = 0;
  for (int i = 0; i < n_again; ++i) {
    double e = 0, v = 0;
    if (std::fscanf(f, "%lf %lf", &e, &v) != 2) { break; }
    es.push_back(real_t(e));
    vs.push_back(real_t(v) * units::barn<real_t>());
    ++count;
  }
  std::fclose(f);
  return count > 0;
}

/// Loads the given Z values from `dir` (the epics2017/phot directory).
/// Returns false if any file is missing, so a bad data path fails loudly rather than
/// silently producing zero cross sections.
template <typename real_t>
__host__ inline bool load_photoelectric(const std::string& dir, const int* zs, int n_z,
                                        PhotoElectricTable<real_t>& out,
                                        std::vector<real_t>& table_e,
                                        std::vector<real_t>& table_v) {
  for (int i = 0; i <= 100; ++i) { out.z_to_index[i] = -1; }
  out.n_elements = 0;
  table_e.clear();
  table_v.clear();

  for (int i = 0; i < n_z; ++i) {
    const int z = zs[i];
    if (z < 1 || z > 100 || out.n_elements >= kPeMaxElements) { return false; }
    if (out.z_to_index[z] >= 0) { continue; }  // already loaded

    PhotoElectricElement<real_t>& e = out.elements[out.n_elements];
    e.z = z;
    int ns_high = 0, ns_low = 0;
    const std::string zs_str = std::to_string(z);
    if (!load_param_file<real_t>(dir + "/pe-high-" + zs_str + ".dat", e.param_high, ns_high)) {
      std::printf("photoelectric: cannot read pe-high-%d.dat in %s\n", z, dir.c_str());
      return false;
    }
    if (!load_param_file<real_t>(dir + "/pe-low-" + zs_str + ".dat", e.param_low, ns_low)) {
      std::printf("photoelectric: cannot read pe-low-%d.dat in %s\n", z, dir.c_str());
      return false;
    }
    if (ns_high != ns_low) { return false; }
    e.n_shells = ns_high;

    load_cs_file<real_t>(dir + "/pe-cs-" + zs_str + ".dat", table_e, table_v, e.cs_offset,
                         e.cs_count);
    load_cs_file<real_t>(dir + "/pe-le-cs-" + zs_str + ".dat", table_e, table_v, e.le_offset,
                         e.le_count);

    // Per-shell cross sections: n_shells blocks of "emin emax npoints shell_id" then points.
    e.n_ss = 0;
    {
      FILE* fs = std::fopen((dir + "/pe-ss-cs-" + zs_str + ".dat").c_str(), "r");
      if (fs != nullptr) {
        for (int s = 0; s < e.n_shells && s < kPeMaxShells; ++s) {
          double emin = 0, emax = 0;
          int np = 0, sid = 0;
          if (std::fscanf(fs, "%lf %lf %d %d", &emin, &emax, &np, &sid) != 4) { break; }
          e.ss_offset[s] = static_cast<int>(table_e.size());
          e.ss_id[s] = sid;
          int got = 0;
          for (int k = 0; k < np; ++k) {
            double ee = 0, vv = 0;
            if (std::fscanf(fs, "%lf %lf", &ee, &vv) != 2) { break; }
            table_e.push_back(real_t(ee));
            table_v.push_back(real_t(vv) * units::barn<real_t>());
            ++got;
          }
          e.ss_count[s] = got;
          ++e.n_ss;
        }
        std::fclose(fs);
      }
    }

    out.z_to_index[z] = out.n_elements;
    ++out.n_elements;
  }
  out.table_n = static_cast<int>(table_e.size());
  return true;
}

// ---------------------------------------------------------------- device evaluation

/// Linear interpolation in a shared (E, value) block, which is what Geant4 does here too -
/// these vectors are built with spline disabled in the epics2017 configuration. See the file
/// header.
template <typename real_t>
__host__ __device__ inline real_t interp_block(const real_t* es, const real_t* vs, int off,
                                               int n, real_t e) {
  if (n <= 0) { return real_t(0); }
  if (e <= es[off]) { return vs[off]; }
  if (e >= es[off + n - 1]) { return vs[off + n - 1]; }
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (es[off + mid] <= e) { lo = mid; } else { hi = mid; }
  }
  const real_t e0 = es[off + lo], e1 = es[off + hi];
  const real_t d = e1 - e0;
  if (d <= real_t(0)) { return vs[off + lo]; }
  const real_t f = (e - e0) / d;
  return vs[off + lo] * (real_t(1) - f) + vs[off + hi] * f;
}

/// Photoelectric cross section per atom, mm^2.
/// Transcribed from G4LivermorePhotoElectricModel::ComputeCrossSectionPerAtom.
template <typename real_t>
__host__ __device__ inline real_t photoelectric_xs_per_atom(const PhotoElectricTable<real_t>& t,
                                                            int z, real_t energy) {
  if (z < 1 || z > 100) { return real_t(0); }
  const int i = t.z_to_index[z];
  if (i < 0) { return real_t(0); }
  const PhotoElectricElement<real_t>& e = t.elements[i];

  const int idx = e.n_shells * 7 - 5;
  // G4: energy = max(energy, paramHigh[idx-1]) - the last shell's binding energy.
  energy = fmax(energy, e.param_high[idx - 1]);
  const real_t x1 = real_t(1) / energy;
  const real_t x2 = x1 * x1;
  const real_t x3 = x2 * x1;

  if (energy >= e.param_high[0]) {
    const real_t x4 = x2 * x2, x5 = x4 * x1;
    return x1 * (e.param_high[idx] + x1 * e.param_high[idx + 1] + x2 * e.param_high[idx + 2]
                 + x3 * e.param_high[idx + 3] + x4 * e.param_high[idx + 4]
                 + x5 * e.param_high[idx + 5]);
  }
  if (energy >= e.param_low[0]) {
    const real_t x4 = x2 * x2, x5 = x4 * x1;
    return x1 * (e.param_low[idx] + x1 * e.param_low[idx + 1] + x2 * e.param_low[idx + 2]
                 + x3 * e.param_low[idx + 3] + x4 * e.param_low[idx + 4]
                 + x5 * e.param_low[idx + 5]);
  }
  if (energy >= e.param_high[1]) {
    return x3 * interp_block(t.table_e, t.table_v, e.cs_offset, e.cs_count, energy);
  }
  return x3 * interp_block(t.table_e, t.table_v, e.le_offset, e.le_count, energy);
}

// ---------------------------------------------------------------- final state

template <typename real_t>
struct PhotoElectricResult {
  Vec3<real_t> electron_dir;
  real_t electron_ekin;
  real_t local_deposit;  ///< the shell binding energy, or the whole photon below threshold
  bool electron_produced;
};

/// Sauter-Gavrila photoelectron direction, transcribed from
/// G4SauterGavrilaAngularDistribution::SampleDirection.
///
/// Note it uses the *photon* energy, not the photoelectron energy: the model's second
/// parameter is unnamed and unused in Geant4. Transcribed as-is rather than "corrected".
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> sauter_gavrila_direction(real_t gamma_energy,
                                                                 const Vec3<real_t>& gamma_dir,
                                                                 Rng& rng) {
  constexpr real_t emin = real_t(1e-6);   // 1 eV
  constexpr real_t emax = real_t(100.0);  // 100 MeV
  const real_t energy = fmax(gamma_energy, emin);
  if (energy > emax) { return gamma_dir; }

  const real_t me = units::electron_mass_c2<real_t>();
  const real_t tau = energy / me;
  const real_t gamma = real_t(1) + tau;
  const real_t beta = sqrt(tau * (tau + real_t(2))) / gamma;
  const real_t ac = (real_t(1) - beta) / beta;
  const real_t a1 = real_t(0.5) * beta * gamma * tau * (gamma - real_t(2));
  const real_t a2 = ac + real_t(2);
  const real_t gtmax = real_t(2) * (a1 + real_t(1) / ac);

  real_t tsam = 0, gtr = 0;
  for (int guard = 0; guard < 1000; ++guard) {
    const real_t r = rng.uniform();
    tsam = real_t(2) * ac * (real_t(2) * r + a2 * sqrt(r)) / (a2 * a2 - real_t(4) * r);
    gtr = (real_t(2) - tsam) * (a1 + real_t(1) / (ac + tsam));
    if (rng.uniform() * gtmax <= gtr) { break; }
  }
  const real_t costheta = real_t(1) - tsam;
  const real_t sint = sqrt(fmax(real_t(0), tsam * (real_t(2) - tsam)));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  const Vec3<real_t> local{sint * cos(phi), sint * sin(phi), costheta};
  return normalize(rotate_uz(local, gamma_dir));
}

/// Samples one photoelectric absorption. The photon is always consumed.
///
/// Shell selection follows G4LivermorePhotoElectricModel::SampleSecondaries: draw against
/// the total parameterized cross section, then walk shells accumulating their individual
/// parameterizations. With fluorescence disabled - the G4EmStandardPhysics default - the
/// shell binding energy is deposited locally, which is exactly what Geant4 does.
template <typename real_t, typename Rng>
__host__ __device__ inline PhotoElectricResult<real_t> sample_photoelectric(
    const PhotoElectricTable<real_t>& t, int z, real_t gamma_energy,
    const Vec3<real_t>& gamma_dir, Rng& rng) {
  PhotoElectricResult<real_t> out{gamma_dir, real_t(0), gamma_energy, false};
  if (z < 1 || z > 100) { return out; }
  const int ei = t.z_to_index[z];
  if (ei < 0) { return out; }
  const PhotoElectricElement<real_t>& e = t.elements[ei];

  const int nn = e.n_shells;
  int shell = 0;
  if (nn > 1) {
    const real_t x1 = real_t(1) / gamma_energy;
    const real_t x2 = x1 * x1, x3 = x2 * x1, x4 = x3 * x1, x5 = x4 * x1;
    const int idx0 = nn * 7 - 5;
    const real_t* p = nullptr;
    if (gamma_energy >= e.param_high[0]) {
      p = e.param_high;
    } else if (gamma_energy >= e.param_low[0]) {
      p = e.param_low;
    }
    if (p != nullptr) {
      const real_t cs0 = rng.uniform()
                         * (p[idx0] + x1 * p[idx0 + 1] + x2 * p[idx0 + 2] + x3 * p[idx0 + 3]
                            + x4 * p[idx0 + 4] + x5 * p[idx0 + 5]);
      for (shell = 0; shell < nn; ++shell) {
        const int idx = shell * 7 + 2;
        if (gamma_energy > p[idx - 1]) {
          const real_t cs = p[idx] + x1 * p[idx + 1] + x2 * p[idx + 2] + x3 * p[idx + 3]
                            + x4 * p[idx + 4] + x5 * p[idx + 5];
          if (cs >= cs0) { break; }
        }
      }
      if (shell >= nn) { shell = nn - 1; }
    } else if (e.n_ss > 0) {
      // Below both parameterizations Geant4 walks the per-shell tabulated cross sections,
      // subtracting each from a uniform draw against the total until it goes negative.
      // Transcribed from the third branch of SampleSecondaries.
      real_t cs = rng.uniform();
      cs *= (gamma_energy >= e.param_high[1])
                ? interp_block(t.table_e, t.table_v, e.cs_offset, e.cs_count, gamma_energy)
                : interp_block(t.table_e, t.table_v, e.le_offset, e.le_count, gamma_energy);
      for (int j = 0; j < e.n_ss; ++j) {
        shell = e.ss_id[j];
        if (shell < 0 || shell >= nn) { shell = (nn > 0) ? nn - 1 : 0; }
        if (gamma_energy > e.param_low[7 * shell + 1]) {
          cs -= interp_block(t.table_e, t.table_v, e.ss_offset[j], e.ss_count[j], gamma_energy);
        }
        if (cs <= real_t(0) || j + 1 == e.n_ss) { break; }
      }
    } else {
      shell = nn - 1;
    }
  }

  const real_t binding = e.param_high[shell * 7 + 1];
  if (gamma_energy < binding) {
    out.local_deposit = gamma_energy;  // G4 deposits the whole photon
    return out;
  }
  out.electron_ekin = gamma_energy - binding;
  out.electron_dir = sauter_gavrila_direction(gamma_energy, gamma_dir, rng);
  out.local_deposit = binding;  // no fluorescence: binding energy stays here
  out.electron_produced = true;
  return out;
}

}  // namespace g4gpu::data
