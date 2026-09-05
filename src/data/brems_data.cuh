// Seltzer-Berger bremsstrahlung: G4EMLOW brem_SB scaled differential cross sections, plus
// the Gauss-Legendre integrations Geant4 uses to turn them into a restricted radiative
// stopping power and a photon-production cross section.
//
// Transcribed from G4SeltzerBergerModel (the DCS) and G4eBremsstrahlungRelModel (the two
// integrations, ComputeBremLoss and ComputeXSectionPerAtom, and their assembly in
// ComputeDEDXPerVolume and ComputeCrossSectionPerAtom).
//
// Architecture note: the integrals are evaluated on the HOST at initialization onto a
// log-spaced energy grid per material, and the device interpolates. That is exactly what
// Geant4 itself does - it builds dE/dx and lambda G4PhysicsTables at init and interpolates
// them while tracking - so it is not an approximation, just the same design.
//
// br<Z> file layout: "k nx ny", then nx x-nodes (kappa = k/T), then ny y-nodes
// (ln(T/MeV)), then ny rows of nx values.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "core/units.cuh"
#include "physics/em/brems_rel.cuh"
#include "core/vec3.cuh"
#include "data/materials.cuh"

namespace g4gpu::data {

constexpr int kSbMaxX = 64;
constexpr int kSbMaxY = 64;
constexpr int kSbMaxElements = 16;

/// One element's scaled DCS table.
template <typename real_t>
struct SBElement {
  int z;
  int nx, ny;
  real_t x[kSbMaxX];             ///< kappa nodes
  real_t y[kSbMaxY];             ///< ln(T/MeV) nodes
  real_t v[kSbMaxX * kSbMaxY];   ///< v[j*nx + i]
};

template <typename real_t>
struct SBTableSet {
  SBElement<real_t> elements[kSbMaxElements];
  int n_elements;
  int z_to_index[101];
};

// ---------------------------------------------------------------- constants

/// 16 * alpha * r_e^2 / 3
template <typename real_t> __host__ __device__ constexpr real_t brem_factor() {
  return real_t(16.0) * units::fine_structure_const<real_t>() * units::classic_electron_radius<real_t>()
         * units::classic_electron_radius<real_t>() / real_t(3.0);
}

/// 4 * pi * r_e * lambda_C^2, the Migdal constant. Electron Compton length
/// (reduced, hbar/mc) is units::electron_compton_length, 3.8615929818578764e-10 mm - CLHEP's
/// own derived value, not the CODATA one; see the note in core/units.cuh.
template <typename real_t> __host__ __device__ constexpr real_t migdal_constant() {
  constexpr real_t lambda_c = units::electron_compton_length<real_t>();  // mm
  return real_t(4.0) * units::pi<real_t>() * units::classic_electron_radius<real_t>() * lambda_c
         * lambda_c;
}

/// 8-point Gauss-Legendre nodes and weights on [0,1], verbatim from
/// G4eBremsstrahlungRelModel::gXGL / gWGL.
template <typename real_t> __host__ __device__ inline const real_t* gl_nodes() {
  static const real_t x[8] = {real_t(1.98550718e-02), real_t(1.01666761e-01),
                              real_t(2.37233795e-01), real_t(4.08282679e-01),
                              real_t(5.91717321e-01), real_t(7.62766205e-01),
                              real_t(8.98333239e-01), real_t(9.80144928e-01)};
  return x;
}
template <typename real_t> __host__ __device__ inline const real_t* gl_weights() {
  static const real_t w[8] = {real_t(5.06142681e-02), real_t(1.11190517e-01),
                              real_t(1.56853323e-01), real_t(1.81341892e-01),
                              real_t(1.81341892e-01), real_t(1.56853323e-01),
                              real_t(1.11190517e-01), real_t(5.06142681e-02)};
  return w;
}

// ---------------------------------------------------------------- loading

template <typename real_t>
__host__ inline bool load_sb_tables(const std::string& dir, const int* zs, int n_z,
                                    SBTableSet<real_t>& out) {
  for (int i = 0; i <= 100; ++i) { out.z_to_index[i] = -1; }
  out.n_elements = 0;
  for (int i = 0; i < n_z; ++i) {
    const int z = zs[i];
    if (out.z_to_index[z] >= 0) { continue; }
    if (out.n_elements >= kSbMaxElements) { return false; }
    const std::string path = dir + "/br" + std::to_string(z);
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
      std::printf("brems: cannot open %s\n", path.c_str());
      return false;
    }
    SBElement<real_t>& e = out.elements[out.n_elements];
    e.z = z;
    int k = 0, nx = 0, ny = 0;
    if (std::fscanf(f, "%d %d %d", &k, &nx, &ny) != 3 || nx < 2 || ny < 2 || nx > kSbMaxX
        || ny > kSbMaxY) {
      std::fclose(f);
      return false;
    }
    e.nx = nx;
    e.ny = ny;
    for (int a = 0; a < nx; ++a) { double t; std::fscanf(f, "%lf", &t); e.x[a] = real_t(t); }
    for (int b = 0; b < ny; ++b) { double t; std::fscanf(f, "%lf", &t); e.y[b] = real_t(t); }
    for (int j = 0; j < ny; ++j) {
      for (int a = 0; a < nx; ++a) {
        double t = 0;
        if (std::fscanf(f, "%lf", &t) != 1) { std::fclose(f); return false; }
        e.v[j * nx + a] = real_t(t);
      }
    }
    std::fclose(f);
    out.z_to_index[z] = out.n_elements;
    ++out.n_elements;
  }
  return true;
}

// ---------------------------------------------------------------- DCS

/// Bilinear interpolation on the (kappa, lnT) grid, matching G4Physics2DVector::Value with
/// bicubic interpolation off, which is the default.
template <typename real_t>
__host__ __device__ inline real_t sb_value(const SBElement<real_t>& e, real_t x, real_t y) {
  int i = 0, j = 0;
  if (x <= e.x[0]) { i = 0; }
  else if (x >= e.x[e.nx - 1]) { i = e.nx - 2; }
  else { while (i + 2 < e.nx && e.x[i + 1] <= x) { ++i; } }
  if (y <= e.y[0]) { j = 0; }
  else if (y >= e.y[e.ny - 1]) { j = e.ny - 2; }
  else { while (j + 2 < e.ny && e.y[j + 1] <= y) { ++j; } }

  const real_t x0 = e.x[i], x1 = e.x[i + 1];
  const real_t y0 = e.y[j], y1 = e.y[j + 1];
  const real_t tx = (x1 > x0) ? (x - x0) / (x1 - x0) : real_t(0);
  const real_t ty = (y1 > y0) ? (y - y0) / (y1 - y0) : real_t(0);
  const real_t v00 = e.v[j * e.nx + i], v10 = e.v[j * e.nx + i + 1];
  const real_t v01 = e.v[(j + 1) * e.nx + i], v11 = e.v[(j + 1) * e.nx + i + 1];
  return v00 * (real_t(1) - tx) * (real_t(1) - ty) + v10 * tx * (real_t(1) - ty)
         + v01 * (real_t(1) - tx) * ty + v11 * tx * ty;
}

/// Differential cross section d(sigma)/dk per atom, before the Z^2 * gBremFactor scaling.
/// Transcribed from G4SeltzerBergerModel::ComputeDXSectionPerAtom, including the positron
/// suppression factor.
template <typename real_t>
__host__ __device__ inline real_t sb_dcs(const SBElement<real_t>& e, real_t kinetic,
                                         real_t gamma_energy, bool is_positron) {
  if (gamma_energy < real_t(0) || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total = kinetic + me;
  const real_t x = gamma_energy / kinetic;
  const real_t y = log(kinetic);  // kinetic is already in MeV

  const real_t pt2 = kinetic * (kinetic + real_t(2) * me);
  const real_t invb2 = total * total / pt2;
  const real_t val = sb_value(e, x, y);
  constexpr real_t millibarn = real_t(1e-3) * real_t(1e-22);  // mm^2
  real_t dxsec = val * invb2 * millibarn / brem_factor<real_t>();

  if (is_positron) {
    constexpr real_t alpha = units::fine_structure_const<real_t>();
    constexpr real_t exp_limit = real_t(-12.0);  // gExpNumLimit
    const real_t invbeta1 = sqrt(invb2);
    const real_t e2 = kinetic - gamma_energy;
    if (e2 <= real_t(0)) { return real_t(0); }
    const real_t invbeta2 = (e2 + me) / sqrt(e2 * (e2 + real_t(2) * me));
    const real_t dum0 = real_t(2) * units::pi<real_t>() * alpha * real_t(e.z)
                        * (invbeta1 - invbeta2);
    if (dum0 < exp_limit) { return real_t(0); }
    dxsec *= exp(dum0);
  }
  return dxsec;
}

// ---------------------------------------------------------------- integrations

/// Integral of k * dsigma/dk from 0 to tmax, per atom (before Z^2 * gBremFactor).
/// Transcribed from G4eBremsstrahlungRelModel::ComputeBremLoss.
template <typename real_t>
__host__ inline real_t sb_brem_loss(const SBElement<real_t>& e, real_t kinetic, real_t tmax,
                                    real_t density_corr, bool is_positron) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total = kinetic + me;
  const real_t alpha_max = tmax / total;
  const int n_sub = static_cast<int>(20 * alpha_max) + 3;
  const real_t delta = alpha_max / real_t(n_sub);
  const real_t* xgl = gl_nodes<real_t>();
  const real_t* wgl = gl_weights<real_t>();

  real_t alpha_i = real_t(0);
  real_t integ = real_t(0);
  for (int l = 0; l < n_sub; ++l) {
    for (int g = 0; g < 8; ++g) {
      const real_t k = (alpha_i + xgl[g] * delta) * total;
      const real_t dcs = sb_dcs(e, kinetic, k, is_positron);
      integ += wgl[g] * dcs / (real_t(1) + density_corr / (k * k));
    }
    alpha_i += delta;
  }
  integ *= delta * total;
  return fmax(integ, real_t(0));
}

/// Integral of dsigma/dk from tmin up, per atom (before Z^2 * gBremFactor).
/// Transcribed from G4eBremsstrahlungRelModel::ComputeXSectionPerAtom.
template <typename real_t>
__host__ inline real_t sb_xsection(const SBElement<real_t>& e, real_t kinetic, real_t tmin,
                                   real_t density_corr, bool is_positron) {
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t total = kinetic + me;
  const real_t alpha_min = std::log(tmin / total);
  const real_t alpha_max = std::log(kinetic / total);
  if (alpha_min >= alpha_max) { return real_t(0); }
  const int n_sub = static_cast<int>(0.45 * (alpha_max - alpha_min)) + 4;
  const real_t delta = (alpha_max - alpha_min) / real_t(n_sub);
  const real_t* xgl = gl_nodes<real_t>();
  const real_t* wgl = gl_weights<real_t>();

  real_t alpha_i = alpha_min;
  real_t xs = real_t(0);
  for (int l = 0; l < n_sub; ++l) {
    for (int g = 0; g < 8; ++g) {
      const real_t k = std::exp(alpha_i + xgl[g] * delta) * total;
      const real_t dcs = sb_dcs(e, kinetic, k, is_positron);
      xs += wgl[g] * dcs / (real_t(1) + density_corr / (k * k));
    }
    alpha_i += delta;
  }
  xs *= delta;
  return fmax(xs, real_t(0));
}

// ---------------------------------------------------------------- material tables

/// 1 keV to 100 TeV at 40 bins per decade, matching the range G4eBremsstrahlung covers
/// (G4EmParameters::MaxKinEnergy defaults to 100 TeV).
constexpr int kBremsBins = 441;

/// The energy at which G4eBremsstrahlung hands over from Seltzer-Berger to the
/// relativistic model (std::min(SB high limit, 1 GeV) in its Initialise).
template <typename real_t> __host__ __device__ constexpr real_t kSeltzerBergerLimit() {
  return real_t(1000.0);  // MeV
}

/// Per-material restricted radiative dE/dx (MeV/mm) and photon-production cross section
/// (1/mm), on a shared log energy grid. Built on the host, read on the device.
template <typename real_t>
struct BremsTable {
  real_t e_min, e_max, log_e_min, inv_dlog;
  real_t dedx[kMaxMaterials][2][kBremsBins];  ///< [material][is_positron][bin]
  real_t xs[kMaxMaterials][2][kBremsBins];
  int n_materials = kNumMaterials;

  __host__ __device__ real_t lookup(const real_t (*tab)[2][kBremsBins], int mat, bool pos,
                                    real_t e) const {
    const int p = pos ? 1 : 0;
    if (e <= e_min) { return tab[mat][p][0] * e / e_min; }
    if (e >= e_max) { return tab[mat][p][kBremsBins - 1]; }
    const real_t f = (log(e) - log_e_min) * inv_dlog;
    const int i = static_cast<int>(f);
    const real_t frac = f - real_t(i);
    return tab[mat][p][i] * (real_t(1) - frac) + tab[mat][p][i + 1] * frac;
  }
  __host__ __device__ real_t dedx_at(int mat, bool pos, real_t e) const {
    return lookup(dedx, mat, pos, e);
  }
  __host__ __device__ real_t xs_at(int mat, bool pos, real_t e) const {
    return lookup(xs, mat, pos, e);
  }
};

/// Evaluates the integrals on the host and fills the tables, assembling over elements the
/// way ComputeDEDXPerVolume and ComputeCrossSectionPerAtom do (Z^2 * n_atoms * gBremFactor).
template <typename real_t>
__host__ inline void build_brems_tables(const Material<real_t>* mats,
                                        const SBTableSet<real_t>& sb, BremsTable<real_t>& out,
                                        int n_materials = kNumMaterials,
                                        real_t e_min = real_t(1e-3),
                                        real_t e_max = real_t(1e8)) {
  out.e_min = e_min;
  out.e_max = e_max;
  out.log_e_min = std::log(e_min);
  const real_t dlog = (std::log(e_max) - out.log_e_min) / real_t(kBremsBins - 1);
  out.inv_dlog = real_t(1) / dlog;

  out.n_materials = n_materials;
  for (int m = 0; m < n_materials; ++m) {
    const Material<real_t>& mat = mats[m];
    // fDensityFactor = gMigdalConstant * electron density; corr uses the total energy.
    const real_t density_factor = migdal_constant<real_t>() * mat.electron_density;
    for (int p = 0; p < 2; ++p) {
      const bool pos = (p == 1);
      const real_t cut = pos ? mat.cut_gamma : mat.cut_gamma;  // photon production cut
      for (int b = 0; b < kBremsBins; ++b) {
        const real_t T = std::exp(out.log_e_min + dlog * real_t(b));
        const real_t total = T + units::electron_mass_c2<real_t>();
        const real_t density_corr = density_factor * total * total;

        real_t dedx = real_t(0), xs = real_t(0);
        const real_t tmax_loss = std::min(cut, T);
        // G4eBremsstrahlung uses G4SeltzerBergerModel below 1 GeV (LPM off) and
        // G4eBremsstrahlungRelModel above it (LPM on), so switch at the same energy.
        if (T > kSeltzerBergerLimit<real_t>()) {
          for (int ie = 0; ie < mat.n_elements; ++ie) {
            const int z = static_cast<int>(mat.z[ie] + real_t(0.5));
            const real_t n = mat.n_atoms[ie];
            if (tmax_loss > real_t(0)) {
              dedx += n * em::rel_brem_loss_per_atom(mat, z, T, tmax_loss);
            }
            if (cut < T) { xs += n * em::rel_brem_xs_per_atom(mat, z, T, cut, T); }
          }
          out.dedx[m][p][b] = fmax(dedx, real_t(0));
          out.xs[m][p][b] = fmax(xs, real_t(0));
          continue;
        }
        for (int ie = 0; ie < mat.n_elements; ++ie) {
          const int z = static_cast<int>(mat.z[ie] + real_t(0.5));
          const int zi = sb.z_to_index[z];
          if (zi < 0) { continue; }
          const SBElement<real_t>& el = sb.elements[zi];
          const real_t z2n = real_t(z) * real_t(z) * mat.n_atoms[ie];
          if (tmax_loss > real_t(0)) {
            dedx += z2n * sb_brem_loss(el, T, tmax_loss, density_corr, pos);
          }
          if (cut < T) {
            xs += z2n * sb_xsection(el, T, cut, density_corr, pos);
          }
        }
        out.dedx[m][p][b] = fmax(dedx * brem_factor<real_t>(), real_t(0));
        out.xs[m][p][b] = fmax(xs * brem_factor<real_t>(), real_t(0));
      }
    }
  }
}

// ---------------------------------------------------------------- photon sampling

/// Samples the emitted photon energy between @p tmin and @p tmax.
/// Transcribed from G4SeltzerBergerModel::SampleEnergyTransfer (the non-sampling-table
/// path, which is the default): rejection in ln(k^2 + densityCorr) against a majorant taken
/// from the DCS at the lower edge, with Geant4's peak-limit and low-x boosts.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t sample_brems_energy(const SBElement<real_t>& e,
                                                      real_t kinetic, real_t tmin, real_t tmax,
                                                      real_t density_corr, bool is_positron,
                                                      Rng& rng) {
  if (tmin >= tmax || kinetic <= real_t(0)) { return real_t(0); }
  const real_t me = units::electron_mass_c2<real_t>();
  const real_t xmin = log(tmin * tmin + density_corr);
  const real_t xrange = log(tmax * tmax + density_corr) - xmin;
  const real_t y = log(kinetic);
  const real_t x0 = tmin / kinetic;

  real_t vmax = sb_value(e, x0, y) * real_t(1.02);
  constexpr real_t kEPeakLim = real_t(300.0);   // 300 MeV
  constexpr real_t kELowLim = real_t(20e-3);    // 20 keV
  if (!is_positron && x0 < real_t(0.97)
      && (kinetic > kEPeakLim || kinetic < kELowLim)) {
    vmax = fmax(vmax, real_t(1.1) * sb_value(e, real_t(0.97), y));
  }
  if (x0 < real_t(0.05)) { vmax *= real_t(1.2); }

  constexpr real_t alpha_2pi = real_t(2.0) * real_t(3.14159265358979323846)
                               * units::fine_structure_const<real_t>();
  constexpr real_t exp_limit = real_t(-12.0);

  for (int n = 0; n < 100; ++n) {
    const real_t r0 = rng.uniform(), r1 = rng.uniform();
    const real_t k = sqrt(fmax(exp(xmin + r0 * xrange) - density_corr, real_t(0)));
    if (k <= real_t(0) || k > kinetic) { continue; }
    real_t v = sb_value(e, k / kinetic, y);
    if (is_positron) {
      const real_t e1 = kinetic - tmin;
      const real_t invbeta1 = (e1 + me) / sqrt(e1 * (e1 + real_t(2) * me));
      const real_t e2 = kinetic - k;
      if (e2 <= real_t(0)) { continue; }
      const real_t invbeta2 = (e2 + me) / sqrt(e2 * (e2 + real_t(2) * me));
      const real_t dum = alpha_2pi * real_t(e.z) * (invbeta1 - invbeta2);
      v = (dum < exp_limit) ? real_t(0) : v * exp(dum);
    }
    if (v >= vmax * r1) { return k; }
  }
  return real_t(0);
}

/// Modified Tsai cos(theta), transcribed verbatim from G4ModifiedTsai::SampleCosTheta.
/// This is the default angular generator for G4eBremsstrahlungRelModel, which
/// G4SeltzerBergerModel inherits.
template <typename real_t, typename Rng>
__host__ __device__ inline real_t tsai_cos_theta(real_t kinetic, Rng& rng) {
  const real_t uMax = real_t(2) * (real_t(1) + kinetic / units::electron_mass_c2<real_t>());
  constexpr real_t a1 = real_t(1.6);
  constexpr real_t a2 = a1 / real_t(3);
  constexpr real_t border = real_t(0.25);
  real_t u;
  for (int n = 0; n < 1000; ++n) {
    const real_t uu = -log(rng.uniform() * rng.uniform());
    u = (border > rng.uniform()) ? uu * a1 : uu * a2;
    if (u <= uMax) { break; }
  }
  return real_t(1) - real_t(2) * u * u / (uMax * uMax);
}

/// Bremsstrahlung photon direction. Transcribed from G4ModifiedTsai::SampleDirection, which
/// samples from the *pre-emission* electron kinetic energy: the model's second parameter is
/// unnamed and unused in Geant4, exactly as with Sauter-Gavrila.
template <typename real_t, typename Rng>
__host__ __device__ inline Vec3<real_t> sample_brems_direction(real_t electron_kinetic,
                                                               const Vec3<real_t>& dir,
                                                               Rng& rng) {
  const real_t ct = tsai_cos_theta(electron_kinetic, rng);
  const real_t st = sqrt(fmax(real_t(0), (real_t(1) - ct) * (real_t(1) + ct)));
  const real_t phi = units::twopi<real_t>() * rng.uniform();
  return normalize(rotate_uz(Vec3<real_t>{st * cos(phi), st * sin(phi), ct}, dir));
}

}  // namespace g4gpu::data
